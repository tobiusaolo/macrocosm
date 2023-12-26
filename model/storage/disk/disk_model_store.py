import asyncio
import shutil
from model.data import Model, ModelId
from model.storage import utils
from model.storage.local_model_store import LocalModelStore
from transformers import AutoModel, DistilBertModel, DistilBertConfig


class DiskModelStore(LocalModelStore):
    """Local storage based implementation for storing and retrieving a model on disk."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def delete_models(self, hotkey: str):
        """Deletes all models for a given hotkey."""
        shutil.rmtree(
            path=utils.get_local_miner_dir(self.base_dir, hotkey), ignore_errors=True
        )

    def delete_model(self, hotkey: str, model_id: ModelId):
        """Delete the given model."""
        shutil.rmtree(
            path=utils.get_local_model_dir(self.base_dir, hotkey, model_id),
            ignore_errors=True,
        )

    def get_path(self, hotkey: str, model_id: ModelId) -> str:
        """Returns the path to where this store would locate this model."""
        return utils.get_local_model_dir(self.base_dir, hotkey, model_id)

    def store_model(self, hotkey: str, model: Model) -> ModelId:
        """Stores a trained model locally."""

        model.pt_model.save_pretrained(
            save_directory=utils.get_local_model_dir(self.base_dir, hotkey, model.id),
            safe_serialization=True,
        )

        # Return the same model id used as we do not edit the commit information.
        return model.id

    def retrieve_model(self, hotkey: str, model_id: ModelId) -> Model:
        """Retrieves a trained model locally."""

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=utils.get_local_model_dir(
                self.base_dir, hotkey, model_id
            ),
            revision=model_id.commit,
            local_files_only=True,
            use_safetensors=True,
        )

        return Model(id=model_id, pt_model=model)


async def test_roundtrip_model():
    """Verifies that the LocalModelStore can roundtrip a model."""
    model_id = ModelId(
        namespace="TestPath",
        name="TestModel",
        hash="TestHash1",
    )

    pt_model = DistilBertModel(
        config=DistilBertConfig(
            vocab_size=256, n_layers=2, n_heads=4, dim=100, hidden_dim=400
        )
    )

    model = Model(id=model_id, pt_model=pt_model)
    disk_model_store = DiskModelStore("local-models")

    # Clear the local storage
    disk_model_store.delete_model(hotkey="hotkey0", model_id=model_id)

    # Store the model locally.
    disk_model_store.store_model(hotkey="hotkey0", model=model)

    # Retrieve the model locally.
    retrieved_model = disk_model_store.retrieve_model(
        hotkey="hotkey0", model_id=model_id
    )

    # Check that they match.
    # TODO create appropriate equality check.
    print(
        f"Finished the roundtrip and checking that the models match: {str(model) == str(retrieved_model)}"
    )


if __name__ == "__main__":
    asyncio.run(test_roundtrip_model())
