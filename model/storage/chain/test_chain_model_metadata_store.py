import asyncio
import functools
import bittensor as bt
import os
from model.data import ModelId, ModelMetadata
import constants
from model.storage.model_metadata_store import ModelMetadataStore
from typing import Optional

from utilities import utils

from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore


# Can only commit data every ~20 minutes.
async def test_store_model_metadata():
    """Verifies that the ChainModelMetadataStore can store data on the chain."""
    model_id = ModelId(
        namespace="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=net_uid
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(hotkey=hotkey, model_id=model_id)

    print(f"Finished storing {model_id} on the chain.")


async def test_retrieve_model_metadata():
    """Verifies that the ChainModelMetadataStore can retrieve data from the chain."""
    # Uses a hotkey/model from the leaderboard.
    expected_model_id = ModelId(
        namespace="tensorplex-labs",
        name="pretraining-sn9-7B-4",
        commit="6d3c838c6de513b17f882d4421948dcfff2f3be9",
        hash="xGtD6aySF4hdnymo7UY8Aeo1rQbsMTFNQ/xaF0HO7vo=",
    )

    # https://x.taostats.io/extrinsic/2810222-0049
    expected_model_metadata = ModelMetadata(
        id=expected_model_id, block=2810222, extrinisic_index=49
    )

    net_uid = 9
    hotkey_address = "5HimsMbLi1n4t1hpdcmURAn2uBBqFtVKGFS7aDwK4DpB2Tvi"

    subtensor = bt.subtensor()

    # Do not require a wallet for retrieving data.
    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=None, subnet_uid=net_uid
    )

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey_address)

    print(
        f"Expecting matching model metadata: {expected_model_metadata == model_metadata}"
    )


async def test_get_index_in_extrinsics():
    """Verifies that the ChainModelMetadataStore can retrieve extrinsic indices from the chain."""

    # https://x.taostats.io/extrinsic/2810222-0049
    net_uid = 9
    hotkey_address = "5HimsMbLi1n4t1hpdcmURAn2uBBqFtVKGFS7aDwK4DpB2Tvi"

    subtensor = bt.subtensor()

    # Do not require a wallet for retrieving data.
    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=None, subnet_uid=net_uid
    )

    # Retrieve the metadata from the chain.
    index = await metadata_store._get_index_in_extrinsics(hotkey_address, 2810222)

    print(f"Expecting matching index in extrinsics: {49 == index}")


# Can only commit data every ~20 minutes.
async def test_roundtrip_model_metadata():
    """Verifies that the ChainModelMetadataStore can roundtrip data on the chain."""
    model_id = ModelId(
        namespace="TestPath", name="TestModel", hash="TestHash1", commit="1.0"
    )

    # Use a different subnet that does not leverage chain storage to avoid conflicts.
    # TODO switch to a mocked version when it supports commits.
    subtensor = bt.subtensor()

    # Uses .env configured wallet/hotkey/uid for the test.
    coldkey = os.getenv("TEST_COLDKEY")
    hotkey = os.getenv("TEST_HOTKEY")
    net_uid = int(os.getenv("TEST_SUBNET_UID"))

    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    metadata_store = ChainModelMetadataStore(
        subtensor=subtensor, wallet=wallet, subnet_uid=net_uid
    )

    # Store the metadata on chain.
    await metadata_store.store_model_metadata(hotkey=hotkey, model_id=model_id)

    # May need to use the underlying publish_metadata function with wait_for_inclusion: True to pass here.
    # Otherwise it defaults to False and we only wait for finalization not necessarily inclusion.

    # Retrieve the metadata from the chain.
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)

    print(f"Expecting matching metadata: {model_id == model_metadata.id}")


if __name__ == "__main__":
    # Can only commit data every ~20 minutes.
    # asyncio.run(test_roundtrip_model_metadata())
    # asyncio.run(test_store_model_metadata())
    # asyncio.run(test_retrieve_model_metadata())
    asyncio.run(test_get_index_in_extrinsics())
