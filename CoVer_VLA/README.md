
<div align="center">
    <img src="assets/banner.png" height=100 alt="CoVer VLA logo"/>

**Scaling Test-Time Sampling and Verification for Vision-Language-Action Models**

[![arXiv](https://img.shields.io/badge/arXiv-2506.17811-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2506.17811)
[![Project Website](https://img.shields.io/badge/Project-Website-blue?style=for-the-badge)](https://robomonkey-vla.github.io/)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/robomonkey-vla)
[![License](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](LICENSE)

</div>

## Table of contents

- [Setup](#-setup)
- [Run Inference](#-run-inference)
- [Visualize Results](#-visualize-results)
- [Acknowledgements](#-acknowledgements)

---

## 🛠️ Setup

### Prerequisites

- Linux
- Python 3.10
- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) (installed automatically by the setup script)

### Installation

From the **vla-clip** repository root:

```bash
bash CoVer_VLA/scripts/env_simpler_pi.sh
```

This script:

1. Creates virtual environment at `vla-clip/.venv_cover`
2. Installs dependencies from `requirements.txt`
3. Installs SimplerEnv (ManiSkill2, simpler_env) without conflicting deps
4. Installs LeRobot with PI0 support (`lerobot_custom`)
5. Installs Bridge Verifier
6. Installs the inference package
7. Pins PyTorch and Transformers versions

### Activate Environment

```bash
source <vla-clip-root>/.venv_cover/bin/activate
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

---

## 🚀 Run Inference

### Option 1: Run via test script

```bash
cd <vla-clip-root>/CoVer_VLA/inference/experiments/robot/simpler/bashes
./test_pi.sh
```

This runs evaluation on `simpler_widowx` and `simpler_ood` with the verifier enabled.

### Option 2: Run with custom arguments

```bash
cd <vla-clip-root>/CoVer_VLA/inference/experiments/robot/simpler/bashes
python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 100 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_suite_name` | `simpler_widowx` | Task suite: `simpler_widowx`, `simpler_ood`, `simpler_put_eggplant_in_basket`, `simpler_stack_cube`, etc. |
| `--use_verifier` | `True` | Use action verifier to select best action |
| `--policy_batch_inference_size` | `2` | Number of actions sampled per instruction |
| `--lang_rephrase_num` | `8` | Number of language rephrases |
| `--num_trials_per_task` | `300` | Episodes per task |
| `--pretrained_checkpoint` | `juexzz/INTACT-pi0-finetune-rephrase-bridge` | PI0 model (or `juexzz/INTACT-pi0-finetune-bridge`) |

### Baseline without verifier

```bash
python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_widowx \
    --use_verifier False \
    --policy_batch_inference_size 1 \
    --lang_rephrase_num 1
```

---

## 📊 Visualize Results

### Run analysis script

After inference, rollout videos and episode data are saved under:

- `rollouts_openpi_original/` (for `juexzz/INTACT-pi0-finetune-bridge`)
- `rollouts_openpi_rephrase/` (for `juexzz/INTACT-pi0-finetune-rephrase-bridge`)

To generate analysis plots:

```bash
cd <vla-clip-root>/CoVer_VLA/inference/experiments/robot/simpler/bashes
python analyze_success_rate.py --output-dir ./analysis_plots
```

### Generated outputs

| Output | Description |
|--------|-------------|
| `success_rates_all_experiments.png` | Success rates across all experiment types |
| `success_rates_original.png` | In-distribution task success rates |
| `success_rates_rephrase.png` | Rephrase experiment success rates |
| `success_rates_robomonkey.png` | Verifier-based experiment success rates |
| `verifier_scores/` | Verifier score trajectories over time |
| `verifier_distributions/` | Verifier score distribution plots |
| `similarity_vs_time_by_language_folders.png` | Success/failure trajectories by similarity |
| `evaluation_mean_std_*.png` | Per-folder evaluation statistics |

### Analysis options

```bash
python analyze_success_rate.py --output-dir ./analysis_plots
python analyze_success_rate.py --output-dir ./my_plots --include-insufficient
```

- `--output-dir`: Where to save plots (default: `./analysis_plots`)
- `--include-insufficient`: Include tasks with only one experiment type (default: filter out)

---

## 📁 Output Locations

| Output | Path (relative to run directory) |
|--------|----------------------------------|
| Text logs | `experiments/logs/EVAL-<task>-<model>-<timestamp>.txt` |
| Rollout videos (.mp4) | `rollouts_openpi_*/transform_*/lang_*_sample_*/` |
| Episode data (.pkl) | Same as rollout videos |

---

## 📊 Evaluation Results (Reference)

| Task                | Initial Samples | Augmented Samples | Seed 1 | Seed 2 | Seed 3 | Average | Baseline | Success Rate ↑ |
|---------------------|-----------------|-------------------|--------|--------|--------|---------|----------|----------------|
| Eggplant in Basket  | 9               | 32                | 76%    | 66%    | 78%    | 73%     | 54%      | **+19%**       |
| Carrot on Plate     | 5               | 16                | 24%    | 24%    | 26%    | 25%     | 20%      | **+5%**        |
| Spoon on Towel      | 5               | 32                | 46%    | 46%    | 50%    | 47%     | 45%      | **+2%**         |
| Stack Cube          | 9               | 32                | 46%    | 40%    | 48%    | 45%     | 35%      | **+10%**       |

---

## 📚 Acknowledgements

We thank the authors of [OpenVLA](https://github.com/openvla/openvla), [SGLang](https://github.com/sgl-project/sglang), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF), and [OpenVLA-mini](https://github.com/Stanford-ILIAD/openvla-mini) for their contributions.

If you find this project helpful, please consider citing:

```bibtex
@article{kwok25robomonkey,
  title={RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models},
  author={Jacky Kwok and Christopher Agia and Rohan Sinha and Matt Foutter and Shulu Li and Ion Stoica and Azalia Mirhoseini and Marco Pavone},
  journal={arXiv preprint arXiv:2506.17811},
  year={2025},
}
```

---

## 🔎 Troubleshooting

**No Vulkan extensions found:** Run the Vulkan setup script (if available):
```bash
bash scripts/vulkan.sh
```

**Display/OpenGL issues:** Ensure `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa` are set for headless rendering.
