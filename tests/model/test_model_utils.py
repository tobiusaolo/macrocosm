import unittest

from model.utils import (
    get_parameter_limit,
    get_model_size_limit,
    get_model_optimizations,
)
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

    model_optimization_cases = [
        (0, False),
        (2_405_919, False),
        (2_405_920, False),
        (3_405_920, False),
        (BLOCK_7B - 1, False),
        (BLOCK_7B, True),
        (BLOCK_7B + 1, True),
    ]

    def test_get_model_optimizations(self):
        for block, expected_state in self.model_optimization_cases:
            with self.subTest(block=block, expected_state=expected_state):
                assert get_model_optimizations(block) == expected_state


if __name__ == "__main__":
    unittest.main()
