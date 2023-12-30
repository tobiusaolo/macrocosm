import asyncio
import bittensor as bt
import os
from model.data import Model, ModelId
from model.storage.disk import utils
from transformers import AutoModel, DistilBertModel, DistilBertConfig

from model.storage.remote_model_store import RemoteModelStore


class HuggingFaceModelStore(RemoteModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    async def upload_model(self, model: Model, local_path: str) -> ModelId:
        """Uploads a trained model to Hugging Face and also roundtrips it locally."""
        token = os.getenv("HF_ACCESS_TOKEN")
        if not token:
            raise ValueError("No Hugging Face access token found to write to the hub.")

        # PreTrainedModel.save_pretrained only saves locally
        commit_info = model.pt_model.push_to_hub(
            repo_id=model.id.namespace + "/" + model.id.name,
            token=token,
            safe_serialization=True,
        )

        model_id_with_commit = ModelId(
            namespace=model.id.namespace,
            name=model.id.name,
            hash=model.id.hash,
            commit=commit_info.oid,
        )

        # To get the hash we need to redownload it at a provided local_path
        model_with_hash = await self.download_model(model_id_with_commit, local_path)

        # Return a ModelId with both the correct commit and hash.
        return model_with_hash.id

    # TODO actually make this asynchronous with threadpools etc.
    async def download_model(self, model_id: ModelId, local_path: str) -> Model:
        """Retrieves a trained model from Hugging Face."""
        if not model_id.commit:
            raise ValueError("No Hugging Face commit id found to read from the hub.")

        # Transformers library can pick up a model based on the hugging face path (username/model) + rev.
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_id.namespace + "/" + model_id.name,
            revision=model_id.commit,
            cache_dir=local_path,
            use_safetensors=True,
        )

        # Compute the hash of the downloaded model.
        model_hash = utils.get_hash_of_directory(local_path)
        model_id_with_hash = ModelId(
            namespace=model_id.namespace,
            name=model_id.name,
            commit=model_id.commit,
            hash=model_hash,
        )

        return Model(id=model_id_with_hash, pt_model=model)


async def test_roundtrip_model():
    """Verifies that the HuggingFaceModelStore can roundtrip a model in hugging face."""
    hf_name = os.getenv("HF_NAME")
    model_id = ModelId(
        namespace=hf_name,
        name="TestModel",
        hash="TestHash1",
        commit="main",
    )

    pt_model = DistilBertModel(
        config=DistilBertConfig(
            vocab_size=256, n_layers=2, n_heads=4, dim=100, hidden_dim=400
        )
    )

    model = Model(id=model_id, pt_model=pt_model)
    hf_model_store = HuggingFaceModelStore()

    # Store the model in hf getting back the id with commit and hash.
    model.id = await hf_model_store.upload_model(
        model, utils.get_local_model_dir("test-models", "hotkey00", model.id)
    )

    # Retrieve the model from hf. Store at a different hotkey to avoid overwriting.
    retrieved_model = await hf_model_store.download_model(
        model_id=model.id,
        local_path=utils.get_local_model_dir("test-models", "hotkey01", model.id),
    )

    # Check that they match.
    # TODO create appropriate equality check.
    print(
        f"Finished the roundtrip and checking that the models match: {str(model) == str(retrieved_model)}"
    )


async def test_retrieve_model():
    """Verifies that the HuggingFaceModelStore can retrieve a model."""
    model_id = ModelId(
        namespace="pszemraj",
        name="distilgpt2-HC3",
        hash="TestHash1",
        commit="6f9ad47",
    )

    hf_model_store = HuggingFaceModelStore()

    # Retrieve the model from hf (first run) or cache.
    model = await hf_model_store.download_model(
        model_id=model_id,
        local_path=utils.get_local_model_dir("test-models", "hotkey0", model_id),
    )

    print(f"Finished retrieving the model with id: {model.id}")


if __name__ == "__main__":
    asyncio.run(test_retrieve_model())
    asyncio.run(test_roundtrip_model())
