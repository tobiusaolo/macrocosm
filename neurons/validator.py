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

from collections import defaultdict
import datetime as dt
import os
import json
import math
import pickle
import time
import torch
import random
import asyncio
import argparse

import wandb
from model.model_tracker import ModelTracker
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.disk.disk_model_store import DiskModelStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import traceback
import threading
import multiprocessing
from rich.table import Table
from rich.console import Console
from multiprocessing import Value

import bittensor as bt
import pretrain as pt
from utilities.miner_iterator import MinerIterator

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Validator:
    TRACKER_FILENAME = "model_tracker.pickle"
    UIDS_FILENAME = "uids.pickle"

    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device name.",
        )
        parser.add_argument(
            "--wandb.off",
            dest="wandb.on",
            action="store_false",
            help="Turn off wandb logging.",
        )
        parser.add_argument(
            "--blocks_per_epoch",
            type=int,
            default=50,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--pages_per_eval",
            type=int,
            default=3,
            help="Number of pages used to eval each step.",
        )
        parser.add_argument(
            "--sample_min",
            type=int,
            default=30,
            help="Number of uids to eval each step.",
        )
        parser.add_argument(
            "--dont_set_weights",
            action="store_true",
            help="Validator does not set weights on the chain.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Runs steps with max 3 uids to eval for faster testing.",
        )
        parser.add_argument(
            "--model_dir",
            default=os.path.join(pt.ROOT_DIR, "model-store/"),
            help="Where to store downloaded models",
        )
        parser.add_argument(
            "--netuid",
            type=str,
            default=pt.SUBNET_UID,
            help="The subnet UID.",
        )

        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def state_path(self) -> str:
        """
        Constructs a file path for storing validator state.

        Returns:
        str: A string representing the file path.
        """
        return os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                bt.logging.config().logging.logging_dir,
                self.wallet.name,
                self.wallet.hotkey_str,
                self.config.netuid,
                "vali-state",
            )
        )

    def __init__(self):
        self.config = Validator.config()
        bt.logging(config=self.config)

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = self.assert_registered(self.wallet, self.metagraph)

        # Dont log to wandb if offline.
        if not self.config.offline and self.config.wandb.on:
            self.new_wandb_run()

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()

        self.uids_to_eval = []

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.pending_uids_to_eval_lock = threading.Lock()
        self.pending_uids_to_eval = set()

        # Setup a model tracker to track which miner is using which model id.
        self.model_tracker = ModelTracker()

        # Load the state of the validator uids from file.
        uids_filepath = os.path.join(self.state_path(), Validator.UIDS_FILENAME)

        # Load the state of the tracker from file.
        tracker_filepath = os.path.join(self.state_path(), Validator.TRACKER_FILENAME)
        if not os.path.exists(tracker_filepath):
            bt.logging.warning("No tracker state file found. Starting from scratch.")
        else:
            self.model_tracker.load_state(tracker_filepath)

        if not os.path.exists(uids_filepath):
            bt.logging.warning("No uids state file found. Starting from scratch.")
            # === Build initial uids to eval ===
            hotkeys = (
                self.model_tracker.get_miner_hotkey_to_model_metadata_dict().keys()
            )
            uids = []
            for hotkey in hotkeys:
                if hotkey in self.metagraph.hotkeys:
                    uids.append(self.metagraph.hotkeys.index(hotkey))
            self.uids_to_eval = set(uids)
        else:
            with open(uids_filepath, "rb") as f:
                self.uids_to_eval = pickle.load(f)
                self.pending_uids_to_eval = pickle.load(f)

        # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        self.metadata_store = ChainModelMetadataStore(
            self.subtensor, self.wallet, self.config.netuid
        )

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

        # TODO: If self.config.test then do not start these threads and instead do a test_run_step equivalent.

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

        # == Initialize the cleaner thread to remove outdated models ==
        self.clean_thread = threading.Thread(target=self.clean_models, daemon=True)
        self.clean_thread.start()

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()
            self.update_thread.join()
            self.clean_thread.join()

    def assert_registered(self, wallet: bt.wallet, metagraph: bt.metagraph) -> int:
        """Asserts the wallet is a registered validator and returns the validator's UID.

        Raises:
            ValueError: If the wallet is not registered.
        """
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            raise ValueError(
                f"You are not registered. \nUse: \n`btcli s register --netuid {self.config.netuid}` to register via burn \n or btcli s pow_register --netuid {self.config.netuid} to register with a proof of work"
            )
        uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.success(
            f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
        )

        return uid

    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb_run = wandb.init(
            name=run_id,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": pt.__version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

    def save_state(self):
        """Saves the state of the validator to a file."""

        # If we are under test do not save any state.
        if self.config.test:
            return

        bt.logging.trace("Saving validator state.")

        if not os.path.exists(self.state_path()):
            os.makedirs(self.state_path())

        # Save the state of the validator uids to file.
        with open(Validator.UIDS_FILENAME, "wb") as f:
            pickle.dump(self.uids_to_eval, f)
            pickle.dump(self.pending_uids_to_eval, f)

        # Save the state of the tracker to file.
        self.model_tracker.save_state(
            os.path.join(self.state_path(), Validator.TRACKER_FILENAME)
        )

    def update_models(self):
        # Track how recently we updated each uid
        uid_last_checked = dict()

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                bt.logging.trace("Updating models.")
                # Get the next uid to check
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't checked it in the last 5 minutes.
                time_diff = (
                    uid_last_checked[next_uid] - dt.datetime.now()
                    if next_uid in uid_last_checked
                    else None
                )

                if time_diff and time_diff < dt.timedelta(minutes=5):
                    # If we have seen it within 5 minutes then sleep until it has been at least 5 minutes.
                    time_to_sleep = (
                        dt.timedelta(minutes=5) - time_diff
                    ).total_seconds()
                    bt.logging.trace(
                        f"Update loop has already seen this uid in the last 5 minutes. Sleeping {time_to_sleep} seconds."
                    )
                    time.sleep(time_to_sleep)

                uid_last_checked[next_uid] = dt.datetime.now()
                bt.logging.trace(f"Updating model for UID={next_uid}")

                # Get their hotkey from the metagraph.
                hotkey = self.metagraph.hotkeys[next_uid]

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                updated = asyncio.run(self.model_updater.sync_model(hotkey))

                bt.logging.trace(
                    f"Updated model for UID={next_uid}. Was new = {updated}"
                )

                # Ensure we eval the new model on the next loop.
                if updated:
                    with self.pending_uids_to_eval_lock:
                        self.pending_uids_to_eval.add(next_uid)

            except Exception as e:
                bt.logging.error(
                    f"Error in update loop: {e} \n {traceback.format_exc()}"
                )

        bt.logging.info("Exiting update models loop.")

    def clean_models(self):
        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                # Clean out unreferenced models older than 5 mintues.
                hotkey_to_model_metadata = (
                    self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
                )
                hotkey_to_id = {
                    hotkey: metadata.id
                    for hotkey, metadata in hotkey_to_model_metadata.items()
                }
                self.local_store.delete_unreferenced_models(hotkey_to_id, 300)
            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}")

            # Only check every 5 minutes.
            time.sleep(dt.timedelta(minutes=5).total_seconds())

        bt.logging.info("Exiting clean models loop.")

    async def try_set_weights(self, ttl: int):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.weights,
                    wait_for_inclusion=False,
                    version_key=pt.weights_version_key,
                )
            except:
                pass
            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

        try:
            bt.logging.debug(f"Setting weights.")
            await asyncio.wait_for(_try_set_weights(), ttl)
            bt.logging.debug(f"Finished setting weights.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")

    async def try_sync_metagraph(self, ttl: int):
        def sync_metagraph(endpoint):
            metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid)
            metagraph.save()
            self.miner_iterator.set_miner_uids(metagraph.uids.tolist())

        process = multiprocessing.Process(
            target=sync_metagraph, args=(self.subtensor.chain_endpoint,)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
        self.metagraph.load()

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()

        try:
            bt.logging.trace(f"Running step.")
            await asyncio.wait_for(_try_run_step(), ttl)
            bt.logging.trace(f"Finished running step.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to run step after {ttl} seconds")

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        1. Identifies valid models for evaluation (top 30 from last run + newly updated models).
        2. Generates random pages for evaluation and prepares batches for each page from the dataset.
        3. Computes the scoring for each model based on the losses incurred on the evaluation batches.
        4. Calculates wins and win rates for each model to determine their performance relative to others.
        5. Updates the weights of each model based on their performance and applies a softmax normalization.
        6. Implements a blacklist mechanism to remove underperforming models from the evaluation set.
        7. Logs all relevant data for the step, including model IDs, pages, batches, wins, win rates, and losses.
        """

        # Pull relevant uids for step. If they aren't found in the model tracker on eval they will be skipped.
        uids = list(self.uids_to_eval)

        # Add uids with newly updated models to the upcoming batch of evaluations.
        with self.pending_uids_to_eval_lock:
            self.uids_to_eval.update(self.pending_uids_to_eval)
            self.pending_uids_to_eval.clear()

        if not uids:
            bt.logging.debug(
                "No uids to eval. Waiting 5 minutes to download some models."
            )
            time.sleep(300)
            return

        # Keep track of which block this uid last updated their model.
        # Default to an infinite block if we can't retrieve the metadata for the miner.
        uid_to_block = defaultdict(math.inf)

        # Generate random pages for evaluation and prepare batches for each page
        # the dataset contains >900 million pages to eval over.
        pages = [
            random.randint(1, pt.dataset.SubsetFalconLoader.max_pages)
            for _ in range(self.config.pages_per_eval)
        ]
        batches = list(
            pt.dataset.SubsetFalconLoader(
                batch_size=pt.batch_size,
                sequence_length=pt.sequence_length,
                pages=pages,
            )
        )

        # Compute model losses on batches.
        bt.logging.debug(f"computing losses on {uids}")
        losses_per_uid = {muid: None for muid in uids}
        for uid_i in uids:
            # Check that the model is in the tracker.
            hotkey = self.metagraph.hotkeys[uid_i]
            model_i_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                hotkey
            )

            losses = [math.inf for _ in batches]

            if model_i_metadata != None:
                try:
                    # Update the block this uid last updated their model.
                    uid_to_block[uid_i] = model_i_metadata.block

                    # Get the model locally and evaluate its loss.
                    model_i = self.local_store.retrieve_model(
                        hotkey, model_i_metadata.id
                    )

                    losses = pt.validation.compute_losses(
                        model_i, batches, device=self.config.device
                    )

                    del model_i
                except Exception as e:
                    bt.logging.error(
                        f"Error in eval loop: {e}. Setting losses for uid: {uid_i} to infinity."
                    )

            losses_per_uid[uid_i] = losses
            average_model_loss = sum(losses) / len(losses)
            bt.logging.debug(
                f"Compute model losses for uid:{uid_i} with average loss: {average_model_loss}"
            )

        # Compute wins and win rates per uid.
        wins, win_rate = pt.validation.compute_wins(
            uids, losses_per_uid, batches, uid_to_block
        )

        # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(
            [win_rate[uid] for uid in uids], dtype=torch.float32
        )
        step_weights = torch.softmax(model_weights / pt.temperature, dim=0)
        bt.logging.success(f"Computed model wins : {wins}")

        # Update weights based on moving average.
        new_weights = torch.zeros_like(self.weights)
        for i, uid_i in enumerate(uids):
            new_weights[uid_i] = step_weights[i]
        new_weights /= new_weights.sum()
        self.weights = pt.alpha * self.weights + (1 - pt.alpha) * new_weights
        self.weights = self.weights.nan_to_num(0.0)

        # Filter based on win rate removing all by the sample_min best models for evaluation.
        self.uids_to_eval = set(
            sorted(win_rate, key=win_rate.get, reverse=True)[: self.config.sample_min]
        )

        # Log to screen and wandb.
        self.log_step(
            uids,
            uid_to_block,
            pages,
            batches,
            wins,
            win_rate,
            losses_per_uid,
        )
        bt.logging.debug("Finished run step.")

    def log_step(
        self, uids, uid_to_block, pages, batches, wins, win_rate, losses_per_uid
    ):
        # Build step log
        step_log = {
            "timestamp": time.time(),
            "pages": pages,
            "uids": uids,
            "uid_data": {},
        }
        for i, uid in enumerate(uids):
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": uid_to_block[i],
                "average_loss": sum(losses_per_uid[uid]) / len(batches),
                "win_rate": win_rate[uid],
                "win_total": wins[uid],
                "weight": self.weights[uid].item(),
            }
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_loss", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("block", style="magenta")
        for uid in uids:
            try:
                table.add_row(
                    str(uid),
                    str(round(step_log["uid_data"][str(uid)]["average_loss"], 4)),
                    str(round(step_log["uid_data"][str(uid)]["win_rate"], 4)),
                    str(step_log["uid_data"][str(uid)]["win_total"]),
                    str(round(self.weights[uid].item(), 4)),
                    str(step_log["uid_data"][str(uid)]["block"]),
                )
            except:
                pass
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.trace(f"Step results: {step_log}")

        if self.config.wandb.on and not self.config.offline:
            original_format_json = json.dumps(step_log)
            uids = step_log["uids"]
            uid_data = step_log["uid_data"]

            # Create a new dictionary with the required format
            graphed_data = {
                "time": time.time(),
                "block": self.metagraph.block.item(),
                "uid_data": {
                    str(uid): uid_data[str(uid)]["average_loss"] for uid in uids
                },
                "weight_data": {str(uid): self.weights[uid].item() for uid in uids},
            }
            bt.logging.trace("Logging to Wandb")
            self.wandb_run.log(
                {**graphed_data, "original_format_json": original_format_json},
                step=self.global_step,
            )
            bt.logging.trace("finished log to Wandb")

    async def run(self):
        while True:
            try:
                while (
                    self.metagraph.block.item() - self.last_epoch
                    < self.config.blocks_per_epoch
                ):
                    await self.try_run_step(ttl=60 * 20)
                    await self.try_sync_metagraph(ttl=60)
                    self.save_state()
                    bt.logging.debug(
                        f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch."
                    )
                    self.global_step += 1

                if (
                    not self.config.dont_set_weights
                    and not self.config.offline
                    and not self.config.test
                ):
                    await self.try_set_weights(ttl=60)
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info(
                    "KeyboardInterrupt caught, gracefully closing the wandb run..."
                )
                self.save_state()
                if self.wandb_run:
                    self.wandb_run.finish()
                exit()

            except Exception as e:
                self.save_state()
                bt.logging.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )


if __name__ == "__main__":
    asyncio.run(Validator().run())
