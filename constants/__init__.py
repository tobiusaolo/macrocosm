from pathlib import Path

# ---------------------------------
# Project Constants.
# ---------------------------------

# The validator WANDB project.
WANDB_PROJECT = "pretraining-subnet"
# The uid for this subnet.
SUBNET_UID = 9
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo (1 Gigabyte).
MAX_HUGGING_FACE_BYTES = 1 * 1024 * 1024 * 1024
# The maximum parameter size allowed for models.
MAX_MODEL_PARAMETER_SIZE = 122268040
# The number of run steps to log to single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = 2002

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.04
# validator score boosting for earlier models.
timestamp_epsilon = 0.01
# validators number of pages to eval over miners on each step.
n_eval_pages = 3
# validator eval batch size.
batch_size = 1
# validator eval sequence length.
sequence_length = 1024
