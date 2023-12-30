# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import math
import os
import wandb
import torch
import random
import argparse
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import pretrain as pt
import bittensor as bt
from transformers import PreTrainedModel
from pretrain.mining import Actions
from utilities import utils
import datetime as dt

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.

    The configuration is responsible for setting up the environment including
    the model path, device to use, and the bittensor wallet and logging configurations.

    Returns:
        A namespace object containing the configuration parameters.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not send model to wandb, does not check if registered",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="The wandb project to log to."
    )
    parser.add_argument("--wandb_entity", type=str, help="The wandb entity to log to.")
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/pretraining",
    )
    parser.add_argument(
        "--avg_loss_upload_threshold",
        type=float,
        default=0,  # Default to never uploading.
        help="The threshold for avg_loss the model must achieve to upload it to hugging face. A miner can only advertise one model, so it should be the best one.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(pt.ROOT_DIR, "local-models/"),
        help="Where to download/save models for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device on which to run. cpu or cuda",
    )
    parser.add_argument(
        "--load_best",
        action="store_true",
        help="If set, the miner loads the best model from wandb to train off.",
    )
    parser.add_argument(
        "--load_uid",
        type=int,
        default=None,
        help="If passed loads the model under the specified uid.",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a model from the specified directory",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=-1,
        help="Number of training epochs (-1 is infinite)",
    )
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate.")
    parser.add_argument("--bs", type=int, default=pt.batch_size, help="Batch size")
    parser.add_argument(
        "--sl", type=int, default=pt.sequence_length, help="Sequence length"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=5,
        help="The number of training accumulation steps.",
    )
    parser.add_argument(
        "--pages_per_epoch",
        type=int,
        default=10,
        help="Number of pages trained on per epoch",
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID.

    Raises:
        ValueError: If the wallet is not registered.
    """
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {pt.SUBNET_UID}` to register via burn \n or btcli s pow_register --netuid {pt.SUBNET_UID} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(
        f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
    )

    return uid


async def load_starting_model(
    actions: Actions, config: bt.config, metagraph: bt.metagraph, model_dir: str
) -> PreTrainedModel:
    """Loads the model to train based on the provided config."""

    # Initialize the model based on the best on the network.
    if config.load_best:
        # Get the best UID be incentive and load it.
        best_uid = pt.graph.best_uid(metagraph)
        model = await actions.load_remote_model(best_uid, metagraph, model_dir)
        bt.logging.success(f"Training with best uid: {best_uid}")
        return model

    # Initialize the model based on a passed uid.
    if config.load_uid is not None:
        # Sync the state from the passed uid.
        model = await actions.load_remote_model(config.load_uid, metagraph, model_dir)
        bt.logging.success(f"Training with model from uid: {config.load_uid}")
        return model

    # Check if we should load a model from a local directory.
    if config.load_model_dir:
        model = actions.load_local_model(config.load_model_dir)
        bt.logging.success("Training with model from disk")
        return model

    # Start from scratch.
    bt.logging.success("Training from scratch.")
    return pt.model.get_model()


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(pt.SUBNET_UID)

    # If running online, make sure the miner is registered, has a hugging face access token, and has provided a repo id.
    my_uid = None
    repo_namespace = None
    repo_name = None
    if not config.offline:
        my_uid = assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()
        repo_namespace, repo_name = utils.validate_hf_repo_id(config.hf_repo_id)

    # Configure the stores and miner actions.
    remote_model_store = HuggingFaceModelStore()
    chain_model_store = ChainModelMetadataStore(subtensor, wallet)
    miner_actions = pt.mining.Actions(
        wallet, repo_namespace, repo_name, chain_model_store, remote_model_store
    )

    # Create a unique run id for this run.
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = pt.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_path, exist_ok=True)

    use_wandb = False
    if not config.offline:
        if config.wandb_project is None or config.wandb_entity is None:
            bt.logging.error(
                "Wandb project or entity not specified. This run will not be logged to wandb"
            )
        else:
            use_wandb = True

    # Init model.
    model: PreTrainedModel = await load_starting_model(
        miner_actions, config, wallet, metagraph
    )
    model = model.train()
    model = model.to(config.device)

    bt.logging.success(f"Saving model to path: {model_path}.")
    miner_actions.save(model, model_path)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    wandb_run = None

    # If using wandb, start a new run.
    if use_wandb:
        wandb_run = wandb.init(
            name=run_id,
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={
                "uid": my_uid,
                "hotkey": wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": pt.__version__,
                "type": "miner",
            },
            dir=model_path,
            allow_val_change=True,
        )

        # Push the model to wandb, for debugging purposes only.
        # This is not seen by validators.
        wandb_run.save(glob_str=model_path)
    else:
        bt.logging.warning(
            "Not posting run to wandb. Either --offline is specified or the wandb settings are missing."
        )

    # Start the training loop
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    best_avg_loss = math.inf
    accumulation_steps = config.accumulation_steps

    try:
        while epoch_step < config.num_epochs or config.num_epochs == -1:
            # Initialize loss accumulator for the epoch
            epoch_loss = 0.0

            # Prepare the data loader with random pages for each epoch
            bt.logging.success(
                f"Loading {config.pages_per_epoch} pages for training this epoch"
            )
            random_pages = [
                random.randint(1, pt.dataset.SubsetFalconLoader.max_pages)
                for _ in range(config.pages_per_epoch)
            ]
            loader = pt.dataset.SubsetFalconLoader(
                batch_size=config.bs, sequence_length=config.sl, pages=random_pages
            )

            # Enumerate over the data loader
            n_batches = 0
            optimizer.zero_grad()  # Initialize gradients to zero

            for i, batch in enumerate(loader):
                # Move the input batch to the device
                inputs = batch.to(model.device)

                # Forward pass: compute the model output and loss
                outputs = model(inputs, labels=inputs)

                loss = outputs.loss / accumulation_steps  # Scale loss
                loss.backward()  # Accumulate gradients

                if (i + 1) % accumulation_steps == 0:
                    n_acc_steps += 1
                    optimizer.step()  # Perform a single optimization step
                    optimizer.zero_grad()  # Clear gradients
                    bt.logging.success(
                        f"Step: {n_acc_steps} loss: {outputs.loss.detach().item()}"
                    )
                    if use_wandb:
                        wandb_run.log(
                            {"loss": outputs.loss.detach(), "n_batches": n_batches},
                            step=n_acc_steps,
                        )

                torch.cuda.empty_cache()

                n_batches += 1
                global_step += 1
                epoch_loss += outputs.loss.detach().item()

            # Calculate the average loss for the epoch
            avg_loss = epoch_loss / n_batches

            # Log the average loss for the epoch
            bt.logging.success(f"Epoch: {epoch_step} average loss: {avg_loss}")
            epoch_step += 1

            # Check if the average loss of this epoch is the best we've seen so far
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss  # Update the best average loss

                bt.logging.success(f"New best average loss: {best_avg_loss}.")

                # Save the model to your mining dir.
                bt.logging.success(f"Saving model to path: {model_path}.")
                miner_actions.save(model, model_path)

                # Also upload this better model to wandb, if enabled.
                if use_wandb:
                    wandb_run.save(glob_str=model_path)

        bt.logging.success("Finished training")
        # Push the model to your run.
        if not config.offline:
            if best_avg_loss < config.avg_loss_upload_threshold:
                bt.logging.success(
                    f"Trained model had a best_avg_loss of {best_avg_loss} which is below the threshold of {config.avg_loss_upload_threshold}. Uploading to hugging face. "
                )

                # First, reload the best model from the training run.
                model_to_upload = miner_actions.load_local_model(model_path)
                await miner_actions.push(model_to_upload)
            else:
                bt.logging.success(
                    f"This training run achieved a best_avg_loss={best_avg_loss}, which did not meet the upload threshold. Not uploading to hugging face."
                )
        else:
            bt.logging.success(
                "Not uploading to hugging face because --offline was specified."
            )

    finally:
        # Important step.
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    print(config)

    asyncio.run(main(config))
