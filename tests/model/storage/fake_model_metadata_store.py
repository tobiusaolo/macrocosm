import asyncio
import functools
import bittensor as bt
import os
from model.data import ModelId, ModelMetadata
from model.storage.chain import constants
from model.storage.model_metadata_store import ModelMetadataStore
from typing import Optional

from utilities import utils


class FakeModelMetadataStore(ModelMetadataStore):
    """Fake implementation for storing and retrieving metadata about a model."""

    def __init__(self):
        self.current_block = 1
        self.metadata = dict()

    async def store_model_metadata(self, hotkey: str, model_id: ModelId):
        """Fake stores model metadata for a specific hotkey."""

        model_metadata = ModelMetadata(id=model_id, block=self.current_block)
        self.current_block += 1

        self.metadata[hotkey] = model_metadata

    async def retrieve_model_metadata(self, hotkey: str) -> ModelMetadata:
        """Retrieves model metadata for specific hotkey"""

        return self.metadata[hotkey]
