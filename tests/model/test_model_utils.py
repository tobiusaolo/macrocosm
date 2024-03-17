import unittest

from model.utils import get_parameter_limit, get_model_size_limit
from constants import BLOCK_7B


class TestModelUtils(unittest.TestCase):
    param_limit_cases = [
        (0, 186_000_000),
        (2_405_919, 186_000_000),
        (2_405_920, 772_000_000),
        (3_405_920, 772_000_000),
        (BLOCK_7B - 1, 772_000_000),
        (BLOCK_7B, 7_300_000_000),
        (BLOCK_7B + 1, 7_300_000_000),
    ]

    def test_get_parameter_limit(self):
        for block, expected_limit in self.param_limit_cases:
            with self.subTest(block=block, expected_limit=expected_limit):
                assert get_parameter_limit(block) == expected_limit

    model_size_limit_cases = [
        (0, 5 * 1024 * 1024 * 1024),
        (2_405_919, 5 * 1024 * 1024 * 1024),
        (2_405_920, 5 * 1024 * 1024 * 1024),
        (3_405_920, 5 * 1024 * 1024 * 1024),
        (BLOCK_7B - 1, 5 * 1024 * 1024 * 1024),
        (BLOCK_7B, 15 * 1024 * 1024 * 1024),
        (BLOCK_7B + 1, 15 * 1024 * 1024 * 1024),
    ]

    def test_get_model_size_limit(self):
        for block, expected_limit in self.model_size_limit_cases:
            with self.subTest(block=block, expected_limit=expected_limit):
                assert get_model_size_limit(block) == expected_limit


if __name__ == "__main__":
    unittest.main()
