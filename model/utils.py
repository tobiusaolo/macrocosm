import constants
from model.data import ModelParameters


def get_model_parameters(block: int) -> ModelParameters:
    """Returns the model parameters at block."""
    parameters = None
    for b, params in constants.MODEL_PARAMETERS_BY_BLOCK:
        if block >= b:
            parameters = params
    assert parameters is not None, f"No model parameters found for block {block}"
    return parameters
