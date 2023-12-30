import datetime
import shutil
import time
import unittest
from model.data import ModelId
import model.storage.disk.utils as utils
import os


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.base_dir = "test-models"
        self.sep = os.path.sep

    def tearDown(self):
        shutil.rmtree(path=self.base_dir, ignore_errors=True)

    def test_get_local_miners_dir(self):
        miners_dir = utils.get_local_miners_dir(self.base_dir)

        expected_path = self.base_dir + self.sep + "models"
        self.assertEqual(miners_dir, expected_path)

    def test_get_local_miner_dir(self):
        hotkey = "test-hotkey"

        miner_dir = utils.get_local_miner_dir(self.base_dir, hotkey)

        expected_path = self.base_dir + self.sep + "models" + self.sep + hotkey
        self.assertEqual(miner_dir, expected_path)

    def test_get_local_model_dir(self):
        hotkey = "test-hotkey"
        namespace = "test-namespace"
        name = "test-name"
        commit = "test-commit"
        model_id = ModelId(
            namespace=namespace, name=name, hash="test-hash", commit=commit
        )

        model_dir = utils.get_local_model_dir(self.base_dir, hotkey, model_id)

        expected_path = (
            self.base_dir
            + self.sep
            + "models"
            + self.sep
            + hotkey
            + self.sep
            + namespace
            + "_"
            + name
            + "_"
            + commit
        )
        self.assertEqual(model_dir, expected_path)

    def test_get_newest_datetime_under_path(self):
        base_dir = "test-models"
        file_name = "test.txt"
        path = base_dir + os.path.sep + file_name

        os.mkdir(base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        last_modified_expected = datetime.datetime.fromtimestamp(os.path.getmtime(path))

        last_modified_actual = utils.get_newest_datetime_under_path(base_dir)

        self.assertEqual(last_modified_actual, last_modified_expected)

    def test_get_newest_datetime_under_path_empty(self):
        base_dir = "test-models"

        last_modified_expected = datetime.datetime.max

        last_modified_actual = utils.get_newest_datetime_under_path(base_dir)

        self.assertEqual(last_modified_actual, last_modified_expected)

    def test_remove_dir_out_of_grace(self):
        base_dir = "test-models"
        file_name = "test.txt"
        path = base_dir + os.path.sep + file_name

        os.mkdir(base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        # Sleep to ensure we are out of grace.
        time.sleep(1)

        self.assertTrue(os.path.exists(base_dir))
        utils.remove_dir_out_of_grace(base_dir, 0)
        self.assertFalse(os.path.exists(base_dir))

    def test_remove_dir_out_of_grace_in_grace(self):
        base_dir = "test-models"
        file_name = "test.txt"
        path = base_dir + os.path.sep + file_name

        os.mkdir(base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        self.assertTrue(os.path.exists(base_dir))
        utils.remove_dir_out_of_grace(base_dir, 60)
        self.assertTrue(os.path.exists(base_dir))


if __name__ == "__main__":
    unittest.main()
