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

import os
import time
from typing import Optional
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
import bittensor as bt
from transformers import PreTrainedModel, AutoModelForCausalLM
import pretrain as pt
from safetensors.torch import load_model

from utilities import utils


def model_path(base_dir: str, run_id: str) -> str:
    """
    Constructs a file path for storing the model relating to a training run.
    """
    return os.path.join(base_dir, "training", run_id)


class Actions:
    """A suite of actions for Miners to save/load and upload/download models."""

    def __init__(
        self,
        wallet: bt.wallet,
        hf_repo_namespace: str,
        hf_repo_name: str,
        model_metadata_store: ModelMetadataStore,
        remote_model_store: RemoteModelStore,
    ):
        self.wallet = wallet
        self.hf_repo_namespace = hf_repo_namespace
        self.hf_repo_name = hf_repo_name
        self.model_metadata_store = model_metadata_store
        self.remote_model_store = remote_model_store

    @classmethod
    def create(
        cls,
        config: bt.config,
        wallet: bt.wallet,
        subtensor: Optional[bt.subtensor] = None,
    ):
        subtensor = subtensor or bt.subtensor(config)
        remote_model_store = HuggingFaceModelStore()
        chain_model_store = ChainModelMetadataStore(
            subtensor, wallet, subnet_uid=config.netuid
        )
        repo_namespace, repo_name = utils.validate_hf_repo_id(config.hf_repo_id)

        return Actions(
            wallet, repo_namespace, repo_name, chain_model_store, remote_model_store
        )

    def save(self, model: PreTrainedModel, model_dir: str):
        """Saves a model to the provided directory"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Save the model state to the specified path.
        model.save_pretrained(
            save_directory=model_dir,
            safe_serialization=True,
        )

    def load_gpt2_model(self, model_file: str) -> PreTrainedModel:
        """For loading GPT2 models from the previous version of this subnet."""
        model = pt.model.get_model()
        load_model(model, model_file)
        return model

    def load_local_model(self, model_dir: str) -> PreTrainedModel:
        """Loads a model from a directory."""
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            local_files_only=True,
            use_safetensors=True,
        )

    async def load_remote_model(
        self, uid: int, metagraph: bt.metagraph, download_dir: str
    ) -> PreTrainedModel:
        """Loads the model currently being advertised by the Miner with the given UID.

        Args:
            uid (int): The UID of the Miner who's model should be downloaded.
            metagraph (bt.metagraph): The metagraph of the current subtensor.
            download_dir (str): The directory to download the model to.
        """
        hotkey = metagraph.hotkeys[uid]
        model_metadata = await self.model_metadata_store.retrieve_model_metadata(hotkey)
        if not model_metadata:
            raise ValueError(f"No model metadata found for miner {uid}")

        model: Model = await self.remote_model_store.download_model(
            model_metadata.id, download_dir
        )
        return model.pt_model

    async def push(self, model: PreTrainedModel, retry_delay_secs: int = 60):
        """Pushes the model to Hugging Face and publishes it on the chain for evaluation by validators."""
        bt.logging.info("Pushing model")

        # First upload the model to HuggingFace.
        model_id = ModelId(namespace=self.hf_repo_namespace, name=self.hf_repo_name)
        model_id = await self.remote_model_store.upload_model(
            Model(id=model_id, pt_model=model)
        )

        bt.logging.success(
            f"Uploaded model to hugging face. Now committing to the chain with model_id: {model_id}"
        )

        # We can only commit to the chain every 20 minutes, so run this in a loop, until
        # successful.
        while True:
            try:
                await self.model_metadata_store.store_model_metadata(
                    self.wallet.hotkey.ss58_address, model_id
                )

                bt.logging.success("Committed model to the chain.")
                break
            except Exception as e:
                bt.logging.error(f"Failed to advertise model on the chain: {e}")
                bt.logging.error(f"Retrying in {retry_delay_secs} seconds...")
                time.sleep(retry_delay_secs)
