# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

__version__ = "2.1.0"
from pathlib import Path


version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
weights_version_key = 2002

# validator weight moving average term
alpha = 0.9
# validator scoring exponential temperature
temperature = 0.25
# validator update model timeout (time between checking uids)
update_model_timeout = 2  # 2 = checks all models every 256 * 2 seconds.
# validator score boosting for earlier models.
timestamp_epsilon = 0.01
# validators number of pages to eval over miners on each step.
n_eval_pages = 3
# validator eval batch size.
batch_size = 1
# validator eval sequence length.
sequence_length = 1024
# The validator WANDB project.
WANDB_PROJECT = "pretraining-subnet"
# The uid for this subnet.
SUBNET_UID = 9
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent

from . import dataset
from . import mining
from . import model

# The maximum bytes for the hugging face repo (1 Gigabyte).
MAX_HUGGING_FACE_BYTES = 1 * 1024 * 1024 * 1024
