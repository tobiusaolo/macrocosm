import constants


def get_parameter_limit(block: int) -> int:
    """Returns the maximum number of parameters allowed for a model at block."""
    limit = None
    for b, lim in constants.MAX_MODEL_PARAMETER_SIZES:
        if block >= b:
            limit = lim
    assert limit is not None, f"No parameter limit found for block {block}"
    return limit


def get_model_size_limit(block: int) -> int:
    """Returns the maximum model size allowed for a model at block."""
    limit = None
    for b, lim in constants.MAX_MODEL_BYTES:
        if block >= b:
            limit = lim
    assert limit is not None, f"No model size limit found for block {block}"
    return limit
