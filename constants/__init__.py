from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
)
from model.data import ModelParameters

# ---------------------------------
# Project Constants.
# ---------------------------------

__version__ = "2.2.1"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The validator WANDB project.
WANDB_PROJECT = "pretraining-subnet"
# The uid for this subnet.
SUBNET_UID = 9
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# Block at which 7b models, 2048 sequence lengths, bfloat16, and flash attention are used.
BLOCK_7B = 9_999_999
SEQUENCE_LENGTH_1 = 1024
SEQUENCE_LENGTH_2 = 2048
# A mapping of block numbers to the supported model types as of that block.
ALLOWED_MODEL_TYPES_1 = {
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
}
ALLOWED_MODEL_TYPES_2 = {
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
}
# A mapping of block numbers to ModelParameters.
MODEL_PARAMETERS_BY_BLOCK = [
    (
        0,
        ModelParameters(
            sequence_length=1024,
            optimized=False,
            max_model_bytes=5 * 1024 * 1024 * 1024,
            max_model_parameters=186_000_000,
            allowed_model_types=ALLOWED_MODEL_TYPES_1,
        ),
    ),
    (
        2_405_920,
        ModelParameters(
            sequence_length=1024,
            optimized=False,
            max_model_bytes=5 * 1024 * 1024 * 1024,
            max_model_parameters=772_000_000,
            allowed_model_types=ALLOWED_MODEL_TYPES_1,
        ),
    ),
    (
        BLOCK_7B,
        ModelParameters(
            sequence_length=2048,
            optimized=True,
            max_model_bytes=15 * 1024 * 1024 * 1024,
            max_model_parameters=7_300_000_000,
            allowed_model_types=ALLOWED_MODEL_TYPES_2,
        ),
    ),
]

# The number of run steps to log to single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# validator weight moving average term
alpha = 0.5
# validator scoring exponential temperature
temperature = 0.04
# validator score boosting for earlier models.
timestamp_epsilon = 0.005
# validators number of pages to eval over miners on each step.
n_eval_pages = 6
# validator eval batch size.
batch_size = 1
