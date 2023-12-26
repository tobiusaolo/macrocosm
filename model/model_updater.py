from model.data import Model, ModelId, ModelMetadata
from model.model_tracker import ModelTracker
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
from utils import utils
import functools


class ModelUpdater:
    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteModelStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.model_tracker = model_tracker

    def _get_metadata(self, hotkey: str, ttl: int) -> ModelMetadata:
        """Get metadata in a wrapper with a ttl to ensure we avoid hangs.

        Args:
            hotkey (str): Hotkey of the miner to fetch metadata for.
            ttl (int): How long to wait on reading the metadata.

        Returns:
            ModelMetadata: Metadata about a model.
        """
        partial = functools.partial(self.metadata_store.retrieve_model_metadata, hotkey)
        return utils.run_in_subprocess(partial, ttl)

    def sync_model(self, hotkey: str, local_path: str):
        # Get the metadata for the miner
        metadata = self._get_metadata(hotkey)

        # Check what model id the model tracker currently has for this hotkey
        tracker_model_id = self.model_tracker.get_model_id_for_miner_hotkey(hotkey)
        if metadata.id == tracker_model_id:
            return

        # Otherwise we need to read the new model (which stores locally) based on the metadata.
        self.remote_store.download_model(metadata.id, local_path)
