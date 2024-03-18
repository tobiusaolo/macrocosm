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


def get_model_optimizations(block: int) -> bool:
    """Returns if optimizations should be used for a model at block."""
    use_optimizations = None
    for b, enabled in constants.OPTIMIZATIONS_USED:
        if block >= b:
            use_optimizations = enabled
    assert enabled is not None, f"No optimization usage found for block {block}"
    return use_optimizations


def get_allowed_model_types(block: int) -> dict[type]:
    """Returns the allowed model types for a model at block."""
    model_types = None
    for b, types in constants.ALLOWED_MODEL_TYPES:
        if block >= b:
            model_types = types
    assert model_types is not None, f"No allowed model types found for block {block}"
    return model_types


def get_model_sequence_length(block: int) -> int:
    """Returns the sequence length required for a model at block."""
    length = None
    for b, len in constants.SEQUENCE_LENGTHS:
        if block >= b:
            length = len
    assert length is not None, f"No sequence length found for block {block}"
    return length
