import asyncio
import functools
import bittensor as bt
import os
from model.data import ModelId, ModelMetadata
import constants
from model.storage.model_metadata_store import ModelMetadataStore
from typing import Optional

from utilities import utils


class ChainModelMetadataStore(ModelMetadataStore):
    """Chain based implementation for storing and retrieving metadata about a model."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        wallet: Optional[bt.wallet] = None,
        subnet_uid: int = constants.SUBNET_UID,
        archive_subtensor: bt.subtensor = bt.subtensor(network="archive"),
    ):
        self.subtensor = subtensor
        # Use an 'archive' to ensure we can get extrinsics from older blocks.
        self.archive_subtensor = archive_subtensor
        self.wallet = (
            wallet  # Wallet is only needed to write to the chain, not to read.
        )
        self.subnet_uid = subnet_uid

    def _get_index_in_extrinsics_impl(self, hotkey: str, block: int) -> Optional[int]:
        """Impl to get the index in extrinsics of the set commitment for the given block and hotkey.

        Args:
            hotkey (str): Hotkey that did the set commitment.
            block (int): Block that the set commitment occurred in.
        """
        block_data = self.archive_subtensor.substrate.get_block(None, block)

        if not block_data:
            return None

        # Check each extrinsic in the block for the set_commitment by the provided hotkey.
        # Hotkeys can only set_commitment once every 20 minutes so just take the first we see.
        for idx, extrinsic in enumerate(block_data["extrinsics"]):
            # Check function name first, otherwise it may not have an address.
            if (
                extrinsic["call"]["call_function"]["name"] == "set_commitment"
                and extrinsic["address"] == hotkey
            ):
                return idx

        # This should never happen since we already confirmed there was metadata for this block.
        bt.logging.warning(
            f"Did not find any set_commitment for block {block} by hotkey {hotkey}"
        )
        return None

    async def _get_index_in_extrinsics(self, hotkey: str, block: int) -> Optional[int]:
        """Gets the index in extrinsics of the set commitment for the given block and hotkey.

        Args:
            hotkey (str): Hotkey that did the set commitment.
            block (int): Block that the set commitment occurred in.
        """
        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            self._get_index_in_extrinsics_impl,
            hotkey,
            block,
        )
        subprocess_partial = functools.partial(utils.run_in_subprocess, partial, 60)
        return utils.run_with_retry(subprocess_partial, single_try_timeout=65)

    async def store_model_metadata(self, hotkey: str, model_id: ModelId):
        """Stores model metadata on this subnet for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            self.subtensor.commit,
            self.wallet,
            self.subnet_uid,
            model_id.to_compressed_str(),
        )
        subprocess_partial = functools.partial(utils.run_in_subprocess, partial, 60)
        utils.run_with_retry(subprocess_partial, single_try_timeout=65)

    async def retrieve_model_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Retrieves model metadata on this subnet for specific hotkey"""

        # Wrap calls to the subtensor in a subprocess with a timeout to handle potential hangs.
        partial = functools.partial(
            bt.extrinsics.serving.get_metadata, self.subtensor, self.subnet_uid, hotkey
        )
        subprocess_partial = functools.partial(utils.run_in_subprocess, partial, 60)
        metadata = utils.run_with_retry(subprocess_partial, single_try_timeout=65)

        if not metadata:
            return None

        extrinsic_index = await self._get_index_in_extrinsics(hotkey, metadata["block"])

        if not extrinsic_index:
            bt.logging.warning(
                f"Failed to find extrinsic index for hotkey: {hotkey} at block: {metadata['block']}"
            )
            return None

        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]

        chain_str = bytes.fromhex(hex_data).decode()

        model_id = None

        try:
            model_id = ModelId.from_compressed_str(chain_str)
        except:
            # If the metadata format is not correct on the chain then we return None.
            bt.logging.trace(
                f"Failed to parse the metadata on the chain for hotkey {hotkey}."
            )
            return None

        model_metadata = ModelMetadata(
            id=model_id, block=metadata["block"], extrinisic_index=extrinsic_index
        )

        return model_metadata
