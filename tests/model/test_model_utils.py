import unittest
from model.utils import get_model_parameters
from constants import BLOCK_7B, ALLOWED_MODEL_TYPES_1, ALLOWED_MODEL_TYPES_2
from model.data import ModelParameters


class TestModelUtils(unittest.TestCase):
    MODEL_PARAMETERS_ORIGINAL = ModelParameters(
        sequence_length=1024,
        optimized=False,
        max_model_bytes=5 * 1024 * 1024 * 1024,
        max_model_parameters=186_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_1,
    )
    MODEL_PARAMETERS_772M = ModelParameters(
        sequence_length=1024,
        optimized=False,
        max_model_bytes=5 * 1024 * 1024 * 1024,
        max_model_parameters=772_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_1,
    )
    MODEL_PARAMETERS_7B = ModelParameters(
        sequence_length=2048,
        optimized=True,
        max_model_bytes=15 * 1024 * 1024 * 1024,
        max_model_parameters=7_300_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_2,
    )
    model_parameter_cases = [
        (0, MODEL_PARAMETERS_ORIGINAL),
        (2_405_919, MODEL_PARAMETERS_ORIGINAL),
        (2_405_920, MODEL_PARAMETERS_772M),
        (3_405_920, MODEL_PARAMETERS_772M),
        (BLOCK_7B - 1, MODEL_PARAMETERS_772M),
        (BLOCK_7B, MODEL_PARAMETERS_7B),
        (BLOCK_7B + 1, MODEL_PARAMETERS_7B),
    ]

    def test_get_model_parameters(self):
        for block, expected_params in self.model_parameter_cases:
            with self.subTest(block=block, expected_params=expected_params):
                assert get_model_parameters(block) == expected_params


if __name__ == "__main__":
    unittest.main()
