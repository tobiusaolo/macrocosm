import unittest

from model.utils import (
    get_parameter_limit,
    get_model_size_limit,
    get_model_optimizations,
    get_allowed_model_types,
    get_model_sequence_length,
)
from constants import BLOCK_7B, ALLOWED_MODEL_TYPES_1, ALLOWED_MODEL_TYPES_2


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

    model_types_cases = [
        (0, ALLOWED_MODEL_TYPES_1),
        (2_405_919, ALLOWED_MODEL_TYPES_1),
        (2_405_920, ALLOWED_MODEL_TYPES_1),
        (3_405_920, ALLOWED_MODEL_TYPES_1),
        (BLOCK_7B - 1, ALLOWED_MODEL_TYPES_1),
        (BLOCK_7B, ALLOWED_MODEL_TYPES_2),
        (BLOCK_7B + 1, ALLOWED_MODEL_TYPES_2),
    ]

    def test_get_allowed_model_types(self):
        for block, expected_types in self.model_types_cases:
            with self.subTest(block=block, expected_types=expected_types):
                assert get_allowed_model_types(block) == expected_types

    model_sequence_length_cases = [
        (0, 1024),
        (2_405_919, 1024),
        (2_405_920, 1024),
        (3_405_920, 1024),
        (BLOCK_7B - 1, 2048),
        (BLOCK_7B, 2048),
        (BLOCK_7B + 1, 2048),
    ]

    def test_get_model_sequence_length(self):
        for block, expected_length in self.model_sequence_length_cases:
            with self.subTest(block=block, expected_length=expected_length):
                assert get_model_sequence_length(block) == expected_length


if __name__ == "__main__":
    unittest.main()
