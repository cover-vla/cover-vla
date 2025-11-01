 <b>This README is functionally complete as of 06/26/2025. So if you find something missing, please open an issue and we will take care of it as soon as possible.</b>

:new: [2025-6-26] *Tutorial for Evaluation Updated*

:new: [2025-6-15] *Model Checkpoints Uploaded. Tutorial for Training/Fine-tuning Updated*

:new: [2025-6-12] *Made Public*.

# INT-ACT
### [[Page](https://ai4ce.github.io/INT-ACT/)] | [[Paper](http://arxiv.org/abs/2506.09930)]
This is the official implementation of [From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models](https://ai4ce.github.io/INT-ACT/)

## Table of Contents
- [INT-ACT](#int-act)
    - [\[Page\] | \[Paper\]](#page--paper)
  - [Table of Contents](#table-of-contents)
  - [TODO](#todo)
  - [Installation](#installation)
    - [Install Inference Environment](#install-inference-environment)
      - [Install Inference Client (Simulator) Environment](#install-inference-client-simulator-environment)
      - [(Octo and Magma) Install Inference Server (Policy) Environment](#octo-and-magma-install-inference-server-policy-environment)
  - [Acquire Data for Training/Fine-tuning](#acquire-data-for-trainingfine-tuning)
  - [Acquire Checkpoints for Evaluation](#acquire-checkpoints-for-evaluation)
  - [Train and Fine-tune](#train-and-fine-tune)
  - [Evaluate/Benchmark](#evaluatebenchmark)
  - [How to Set ENV Variables](#how-to-set-env-variables)

## TODO
- [x] Add more complete documentation for training and evaluation. Currently, the code is all there, but the documentation is sparse.

- [x] Release all relevant model checkpoints on HF

## Installation
Install this codebase by first cloning it.
```
git clone --recurse-submodules https://github.com/ai4ce/INT-ACT.git
cd INT-ACT
```
> [!IMPORTANT]
> See the [How to Set ENV Variables](#how-to-set-env-variables) section for setting up the environment variables.

> [!NOTE]
> This codebase relies on [uv](https://docs.astral.sh/uv/) to manage the virtual environments. It's not strictly required, but the authors can only provide support for this environment management system.

Now simply run 
```
uv sync
```

> [!IMPORTANT]
> This only installed the dependency for training and inference server. Full-scale inference requires installing the inference client (simulator) dependencies.

Inference under different environments, such as `Simpler`, `Simpler-ManiSkill3`, `Libero`, or real world requires installing their own dependency in a separate environment. 

> [!NOTE]
> Server refers to the policy ($\pi_0$, Octo, etc). Client refers to simulator (`Maniskill`) and real-world robots. Simulator/Client will feed its observations to the server to retrieve the action to execute.

We do this to allow a server-client architecture that can separate the very different compute demands of doing training vs. running experiments on a robot


### Install Inference Environment 
#### Install Inference Client (Simulator) Environment
This example will use Simpler as an example.

1. Create a separate virtual environment for this simulator.
```
cd src/experiments/envs/simpler
```
```
uv venv --python=3.10 # The version can change to accommodate your simulator's need.
```
2. Activate the inference virtual environment. This is important because we don't want to install the simulator dependencies in the training environment.
```
source .venv/bin/activate
```
3. Install the simulator dependencies using `pyproject.toml`
```
uv pip install -r pyproject.toml
```
#### (Octo and Magma) Install Inference Server (Policy) Environment
Octo and Magma both requires specialized policy environments due to conflicting dependencies.
This example will use Octo as an example.
1. Create a separate virtual environment for this policy.
```
cd src/experiments/policies/octo_policy_server
```
```
uv venv --python=3.10 # The version can change to accommodate your simulator's need.
```
2. Activate the inference virtual environment. This is important because we don't want to install the policy dependencies in the training environment.
```
source .venv/bin/activate
```
3. Install the policy dependencies using `pyproject.toml`
```
uv pip install -r pyproject.toml
```

## Acquire Data for Training/Fine-tuning
For now, we refer you to Allen Ren's [README](https://github.com/allenzren/open-pi-zero)

## Acquire Checkpoints for Evaluation
We released our trained Pi0 variants on Huggingface. You can find them under the [INTACT collection](https://huggingface.co/collections/ai4ce/intact-probing-suite-684e5601e9ed640fdd9b994b). Specifically, they are:

| Model | Notes | Download Link |
| ------ | ------ | ------ |
| Pi0 finetune | Pi0 finetuned on BridgeV2 | [HF hub](https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge)|
| Pi0 finetune rephrase | Pi0 finetuned on BridgeV2 with task paraphrase | [HF hub](https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge)|
| Pi0 scratch | Pi0 trained from scratch on BridgeV2 | [HF hub](https://huggingface.co/juexzz/INTACT-pi0-scratch-bridge)|

You can find the details in each checkpoint's model card.

For convenience, we also include links to the baselines which have been generously provided by their original authors:

| Model | Reference | Download Link |
| ------ | ------ | ------ |
| Magma | Magma: A Foundation Model for Multimodal AI Agents [*CVPR 2025*](https://microsoft.github.io/Magma/) | [HF hub](https://huggingface.co/microsoft/Magma-8B) |
| SpatialVLA | SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model [*RSS 2025*](https://spatialvla.github.io) | [HF hub](https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge)
| Octo models | Octo: An Open-Source Generalist Robot Policy [*RSS 2024*](https://octo-models.github.io) | [Small (HF)](https://huggingface.co/rail-berkeley/octo-small) / [Base (HF)](https://huggingface.co/rail-berkeley/octo-base)


## Train and Fine-tune
The documentation can be found in [doc/training_finetuning.md](doc/training_finetuning.md).


## Evaluate/Benchmark
The documentation can be found in [doc/evaluation.md](doc/evaluation.md).

## How to Set ENV Variables
1. Create a `set_path.sh` file in the project's root directory
2. Fill out the following variables
```
#!/bin/bash
# used to sync the path on HPC with data from collaborators and the model from the baseline directory
# to avoid redundant data download
# training dataset
export VLA_DATA_DIR=

# logging for trained models, logs, etc
export VLA_LOG_DIR=

# WandB
export VLA_WANDB_ENTITY=

# HF cache. TRANSFORMERS_CACHE is deprecated, but still used by some libraries and they themselves a bit confused tbh
export TRANSFORMERS_CACHE=
export HF_HOME=

# SIMPLER
export MS2_REAL2SIM_ASSET_DIR=
export MS_ASSET_DIR=
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# uv (This is optional if you don't mind uv using your home directory, which may not be the case for HPC)
export UV_CACHE_DIR=
export UV_PYTHON_INSTALL_DIR=

# Singularity (This is obviously optional if you don't use Singularity)
export OVERLAY_EXT3=
```
