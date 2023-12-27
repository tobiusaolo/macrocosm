import unittest
from model.data import ModelId
import model.storage.utils as utils
import os


class TestUtils(unittest.TestCase):
    def test_get_local_miner_dir(self):
        base_dir = "test_base_dir"
        hotkey = "test_hotkey"
        sep = os.path.sep

        miner_dir = utils.get_local_miner_dir(base_dir, hotkey)

        expected_path = base_dir + sep + "models" + sep + hotkey
        self.assertEqual(miner_dir, expected_path)

    def test_get_local_model_dir(self):
        base_dir = "test_base_dir"
        hotkey = "test_hotkey"
        namespace = "test_namespace"
        name = "test_name"
        model_id = ModelId(namespace=namespace, name=name, hash="test_hash")
        sep = os.path.sep

        model_dir = utils.get_local_model_dir(base_dir, hotkey, model_id)

        expected_path = (
            base_dir + sep + "models" + sep + hotkey + sep + namespace + sep + name
        )
        self.assertEqual(model_dir, expected_path)


if __name__ == "__main__":
    unittest.main()
