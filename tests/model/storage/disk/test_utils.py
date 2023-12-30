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
        file_name = "test.txt"
        path = self.base_dir + os.path.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        last_modified_expected = datetime.datetime.fromtimestamp(os.path.getmtime(path))

        last_modified_actual = utils.get_newest_datetime_under_path(self.base_dir)

        self.assertEqual(last_modified_actual, last_modified_expected)

    def test_get_newest_datetime_under_path_empty(self):
        last_modified_expected = datetime.datetime.max

        last_modified_actual = utils.get_newest_datetime_under_path(self.base_dir)

        self.assertEqual(last_modified_actual, last_modified_expected)

    def test_remove_dir_out_of_grace(self):
        file_name = "test.txt"
        path = self.base_dir + self.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        # Sleep to ensure we are out of grace.
        time.sleep(1)

        self.assertTrue(os.path.exists(self.base_dir))
        utils.remove_dir_out_of_grace(self.base_dir, 0)
        self.assertFalse(os.path.exists(self.base_dir))

    def test_remove_dir_out_of_grace_in_grace(self):
        file_name = "test.txt"
        path = self.base_dir + self.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        self.assertTrue(os.path.exists(self.base_dir))
        utils.remove_dir_out_of_grace(self.base_dir, 60)
        self.assertTrue(os.path.exists(self.base_dir))

    def test_get_hash_of_file(self):
        file_name = "test.txt"
        path = self.base_dir + self.sep + file_name

        os.mkdir(self.base_dir)
        file = open(path, "w")
        file.write("test text.")
        file.close()

        # Obtained by running sha256sum test.txt
        expected_file_hash = (
            "b57343bc7573a98211894c74aef2be339f8bbb838bce17cf247fa07fb1ef09e0"
        )
        actual_file_hash = utils.get_hash_of_file(path)

        self.assertEqual(actual_file_hash, expected_file_hash)

    def test_get_hash_of_directory(self):
        # Make two sub directories.
        dir_1 = self.base_dir + self.sep + "dir1"
        dir_2 = self.base_dir + self.sep + "dir2"

        # Write the same two files to both sub directories.
        file_name_1 = "test1.txt"
        file_name_2 = "test2.txt"
        path_1_file_1 = dir_1 + os.path.sep + file_name_1
        path_1_file_2 = dir_1 + os.path.sep + file_name_2
        path_2_file_1 = dir_2 + os.path.sep + file_name_1
        path_2_file_2 = dir_2 + os.path.sep + file_name_2

        path_2_file_2 = dir_2 + os.path.sep + file_name_2
        file_paths = [path_1_file_1, path_1_file_2, path_2_file_1, path_2_file_2]

        os.mkdir(self.base_dir)
        os.mkdir(dir_1)
        os.mkdir(dir_2)

        for file_path in file_paths:
            file = open(file_path, "w")
            file.write("test text.")
            file.close()

        # Test that both sub directories have an equal hash.
        dir_1_hash = utils.get_hash_of_directory(dir_1)
        dir_2_hash = utils.get_hash_of_directory(dir_2)
        self.assertEqual(dir_1_hash, dir_2_hash)

        # Test that the hash for the overall directory does not equal the sub directory.
        base_dir_hash = utils.get_hash_of_directory(self.base_dir)
        self.assertNotEqual(base_dir_hash, dir_1_hash)


if __name__ == "__main__":
    unittest.main()
