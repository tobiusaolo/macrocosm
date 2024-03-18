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
# A mapping of block numbers to whether optimizations (bfloat16 and flash attention) are used.
OPTIMIZATIONS_USED = [(0, False), (BLOCK_7B, True)]

# A mapping of block numbers to the max model size as of that block.
MAX_MODEL_BYTES = [
    (0, 5 * 1024 * 1024 * 1024),
    (BLOCK_7B, 15 * 1024 * 1024 * 1024),
]

# A mapping of block numbers to the max model size as of that block.
# This dictionary must remain ordered by key.
MAX_MODEL_PARAMETER_SIZES = [
    (0, 186_000_000),
    (2_405_920, 772_000_000),
    (BLOCK_7B, 7_300_000_000),
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
n_eval_pages = 3
# validator eval batch size.
batch_size = 1
# validator eval sequence length.
sequence_length = 1024
block_7b_sequence_length = 2048
# List of allowed model types.
allowed_model_types = {
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
}
block_7b_allowed_model_types = {
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
}
