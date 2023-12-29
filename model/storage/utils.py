import os
from model.data import ModelId


def get_local_miner_dir(base_dir: str, hotkey: str) -> str:
    return os.path.join(base_dir, "models", hotkey)


def get_local_model_dir(base_dir: str, hotkey: str, model_id: ModelId) -> str:
    return os.path.join(
        get_local_miner_dir(base_dir, hotkey), model_id.namespace, model_id.name
    )
