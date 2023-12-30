from model.data import Model, ModelId
from model.storage.remote_model_store import RemoteModelStore


class FakeRemoteModelStore(RemoteModelStore):
    """Fake implementation for remotely storing and retrieving a model."""

    def __init__(self):
        self.remote_models = dict()

    async def upload_model(self, model: Model, local_path: str) -> ModelId:
        """Fake uploads a model."""

        # Use provided commit rather than generating a new one.
        self.remote_models[model.id] = model

        return model.id

    async def download_model(self, model_id: ModelId, local_path: str) -> Model:
        """Retrieves a trained model from memory."""

        model = self.remote_models[model_id]

        # Store it at the local_path
        model.pt_model.save_pretrained(
            save_directory=local_path,
            safe_serialization=True,
        )

        return model
