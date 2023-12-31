# Miner

Miners train locally and periodically publish models to hugging face and commit the metadata for that model to the Bittensor chain.

The communication between a miner and a validator happens asynchronously chain and therefore Miners do not need to be running continuously. Validators will use whichever metadata was most recently published by the miner to know which model to download from hugging face.

# System Requirements

Miners will need enough disk space to store their model as they work on. Each uploaded model (As of Jan 1st, 2024) may not be more than 1 GB. It is reommended to have at least 25 GB of disk space.

Miners will need enough processing power to train their model. The device the model is trained on is recommended to be a large (>20 GB GPU) but you may use a CPU instead.

# Getting started

## Prerequisites

1. Get a Hugging Face Account: 

Miner and validators use hugging face in order to share model state information. Miners will be uploading to hugging face and therefore must attain a account from [hugging face](https://huggingface.co/) along with a user access token which can be found by following the instructions [here](https://huggingface.co/docs/hub/security-tokens).


2. Clone the repo

```shell
git clone https://github.com/RaoFoundation/pretraining.git
```

3. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

4. Install the requirements. From your virtual environment, run
```shell
cd pretraining
python -m pip install -e .
```

5. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

6. (Optional) Run a Subtensor instance:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```
---

# Running the Miner

The mining script uploads a model to hugging face which will be evaluated by validators.

See [Validator Psuedocode](docs/validator.md#validator) for more information on how they the evaluation occurs.

# Configuration

## Env File

The Miner requires a .env file with your hugging face access token in order to upload models.

Create a `.env` file in the `pretraining` directory and add the following to it:
```py
HF_ACCESS_TOKEN="YOUR_HF_ACCESS_TOKEN"
```

## Flags

The Miner offers some flags to customize properties, such as how to train the model and which hugging face repo to upload to.

You can view the full set of flags by running
```shell
python ./neurons/miner.py -h
```

Testing the training script. Does not require registration or a hugging face account:
```bash
python neurons/miner.py --wallet.name YOUR_WALLET_NAME --wallet.hotkey YOUR_WALLET_HOTKEY --offline
```

Training your model from scratch and posting the model periodically as its loss decreases on Falcon:
```bash
python neurons/miner.py --wallet.name YOUR_WALLET_NAME --wallet.hotkey YOUR_WALLET_HOTKEY --num_epochs 10 --pages_per_epoch 5 --hf_repo_id YOUR_HF_NAMESPACE/YOUR_HF_MODEL
```

Scraping a model from an already running miner on the network by passing its uid before training:
```bash
python neurons/miner.py --wallet.name YOUR_WALLET_NAME --wallet.hotkey YOUR_WALLET_HOTKEY --num_epochs 10 --pages_per_epoch 5 --load_uid UID_TO_LOAD_FROM --hf_repo_id YOUR_HF_NAMESPACE/YOUR_HF_MODEL
```

Loading the best model on the network based on its incentive.
```bash
python neurons/miner.py --wallet.name YOUR_WALLET_NAME --wallet.hotkey YOUR_WALLET_HOTKEY --num_epochs 10 --pages_per_epoch 5 --load_best --hf_repo_id YOUR_HF_NAMESPACE/YOUR_HF_MODEL
```

Pass the `--device` option to select which GPU to run on. 

---

## Manually uploading a model.

In some cases you may have failed to upload a model or wish to upload a model without further training.

# TODO upload_model.py tool
# TODO talk about the miner actions?