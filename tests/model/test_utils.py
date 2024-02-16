import pytest

from model.utils import get_parameter_limit


class TestUtils:
    @pytest.mark.parametrize(
        ["block", "expected_limit"],
        [
            (0, 186_000_000),
            (2_405_919, 186_000_000),
            (2_405_920, 772_000_000),
            (3_405_920, 772_000_000),
        ],
    )
    def test_get_parameter_limit(self, block, expected_limit):
        assert get_parameter_limit(block) == expected_limit


if __name__ == "__main__":
    pytest.main()
