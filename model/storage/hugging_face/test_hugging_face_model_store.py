import asyncio
import os
from model.data import Model, ModelId
from model.storage.disk import utils
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore

from pretrain.model import get_model


async def test_roundtrip_model():
    """Verifies that the HuggingFaceModelStore can roundtrip a model in hugging face."""
    hf_name = os.getenv("HF_NAME")
    model_id = ModelId(
        namespace=hf_name,
        name="TestModel",
    )

    pt_model = get_model()

    model = Model(id=model_id, pt_model=pt_model)
    hf_model_store = HuggingFaceModelStore()

    # Store the model in hf getting back the id with commit and hash.
    model.id = await hf_model_store.upload_model(model)

    # Retrieve the model from hf.
    retrieved_model = await hf_model_store.download_model(
        model_id=model.id,
        local_path=utils.get_local_miner_dir("test-models", "hotkey0"),
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
        local_path=utils.get_local_miner_dir("test-models", "hotkey0"),
    )

    print(f"Finished retrieving the model with id: {model.id}")


async def test_retrieve_oversized_model():
    """Verifies that the HuggingFaceModelStore can raise an exception if the model is too big."""
    model_id = ModelId(
        namespace="microsoft",
        name="phi-2",
        hash="TestHash1",
        commit="d318676",
    )

    hf_model_store = HuggingFaceModelStore()

    try:
        model = await hf_model_store.download_model(
            model_id=model_id,
            local_path=utils.get_local_miner_dir("test-models", "hotkey0"),
        )
    except ValueError as ve:
        print(f"Caught expected exception for downloading too large of a model: {ve}")


if __name__ == "__main__":
    asyncio.run(test_retrieve_model())
    asyncio.run(test_roundtrip_model())
    asyncio.run(test_retrieve_oversized_model())
