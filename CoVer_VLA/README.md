
<div align="center">
    <img src="../assets/inference-teaser-1.png" width="800" alt="CoVer-VLA Inference Pipeline Teaser"/>

**Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Website](https://img.shields.io/badge/Project-Website-blue?style=for-the-badge)](https://cover-vla.github.io)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/stanfordasl/CoVer-BridgeV2)
[![License](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](LICENSE)

</div>

## Table of contents

- [Setup](#-setup)
- [Download Verifier Checkpoint](#-download-verifier-checkpoint)
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

## 📥 Download Verifier Checkpoint

The CoVer action verifier requires a pretrained checkpoint (~312MB). Download it from Hugging Face:

```bash
cd <vla-clip-root>/bridge_verifier
huggingface-cli download stanfordasl/CoVer-BridgeV2 cover_verifier_bridge.pt --local-dir .
cd ..
```

Or with the newer CLI:

```bash
cd <vla-clip-root>/bridge_verifier
hf download stanfordasl/CoVer-BridgeV2 cover_verifier_bridge.pt --local-dir .
cd ..
```

The checkpoint will be saved to `bridge_verifier/cover_verifier_bridge.pt`, which is automatically loaded when running inference with `--use_verifier True`.

**Hugging Face:** [stanfordasl/CoVer-BridgeV2](https://huggingface.co/stanfordasl/CoVer-BridgeV2)

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

CoVer achieves **22% gain** in-distribution and **13% gain** out-of-distribution on the SIMPLER benchmark (8 rephrases, 5 action samples per rephrase, red-teaming instructions).

**In-Distribution (ID):**

| Task | π0 | π0 + CoVer | π0 (rephrase) + CoVer |
|------|-----|------------|----------------------|
| Carrot on Plate | 48 ± 4 | 48 ± 4 | 52 ± 8 |
| Eggplant in Basket | 74 ± 3 | 89 ± 8 | 95 ± 2 |
| Spoon on Towel | 27 ± 4 | 40 ± 6 | 59 ± 5 |
| Block Stacking | 17 ± 1 | 51 ± 4 | 56 ± 0 |
| **Average** | **41.5** | **57.0** | **65.5** |

**Out-of-Distribution (OOD):**

| Task | π0 | π0 + CoVer | π0 (rephrase) + CoVer |
|------|-----|------------|----------------------|
| Redbull on Plate | 6 ± 1 | 51 ± 3 | 46 ± 3 |
| Zucchini on Towel | 30 ± 3 | 41 ± 1 | 55 ± 6 |
| Tennis in Basket | 53 ± 5 | 91 ± 3 | 85 ± 1 |
| **Average** | **29.7** | **61.0** | **62.0** |

See the [main README](../README.md) for PolaRiS and real-world results.

---

## 📚 Acknowledgements

We thank the authors of [LeRobot](https://github.com/huggingface/lerobot), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [CoVer](https://github.com/stanfordasl/CoVer), and related projects for their contributions.

If you find this project helpful, please consider citing:

```bibtex
@article{kwok25cover,
  title={Scaling Verification Can Be More Effective than Scaling Policy Learning for Vision-Language-Action Alignment},
  author={Jacky Kwok and Xilun Zhang and Mengdi Xu and Yuejiang Liu and Azalia Mirhoseini and Chelsea Finn and Marco Pavone},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
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
