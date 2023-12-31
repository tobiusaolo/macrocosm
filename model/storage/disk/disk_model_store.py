import bittensor as bt
from typing import Dict
from model.data import Model, ModelId
from model.storage.disk import utils
from model.storage.local_model_store import LocalModelStore
from transformers import AutoModelForCausalLM
from pathlib import Path


class DiskModelStore(LocalModelStore):
    """Local storage based implementation for storing and retrieving a model on disk."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

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

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=utils.get_local_model_dir(
                self.base_dir, hotkey, model_id
            ),
            revision=model_id.commit,
            local_files_only=True,
            use_safetensors=True,
        )

        return Model(id=model_id, pt_model=model)

    def delete_unreferenced_models(
        self, valid_models_by_hotkey: Dict[str, ModelId], grace_period_seconds: int
    ):
        """Check across all of local storage and delete unreferenced models out of grace period."""
        # Create a set of valid model paths.
        valid_model_paths = set()
        for hotkey, model_id in valid_models_by_hotkey.items():
            valid_model_paths.add(
                utils.get_local_model_dir(self.base_dir, hotkey, model_id)
            )

        # For each hotkey path on disk using listdir to go one level deep.
        miners_dir = Path(utils.get_local_miners_dir(self.base_dir))
        hotkey_subfolder_names = [d.name for d in miners_dir.iterdir() if d.is_dir]

        for hotkey in hotkey_subfolder_names:
            # Reconstruct the path from the hotkey
            hotkey_path = utils.get_local_miner_dir(self.base_dir, hotkey)

            # If it is not in valid_hotkeys and out of grace period remove it.
            if hotkey not in valid_models_by_hotkey:
                bt.logging.trace(
                    f"Removing directory for unreferenced hotkey: {hotkey} if out of grace."
                )
                utils.remove_dir_out_of_grace(hotkey_path, grace_period_seconds)
            else:
                # Check all the model subfolder paths.
                hotkey_dir = Path(hotkey_path)
                model_subfolder_paths = [
                    str(d) for d in hotkey_dir.iterdir() if d.is_dir
                ]

                for model_path in model_subfolder_paths:
                    if model_path not in valid_model_paths:
                        bt.logging.trace(
                            f"Removing directory for unreferenced model at: {model_path} if out of grace."
                        )
                        utils.remove_dir_out_of_grace(model_path, grace_period_seconds)
