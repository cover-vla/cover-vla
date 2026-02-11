# VLA-CLIP

Vision-Language-Action models with instruction verification for robot manipulation.

This repository contains the CoVer (CoVer_VLA) pipeline for evaluating PI0 policies with an action verifier on the SIMPLER benchmark.

## Overview

| Component | Description |
|-----------|-------------|
| **PI0 Policy** | Vision-language-action model (from LeRobot) for action generation |
| **Action Verifier** | Ensemble model that scores action-instruction alignment |
| **SIMPLER Benchmark** | Robot manipulation tasks in simulation |

## Quick Start

### 1. Setup

From the repository root (`vla-clip/`), run the environment setup script:

```bash
bash CoVer_VLA/scripts/env_simpler_pi.sh
```

This script will:
- Install [uv](https://github.com/astral-sh/uv) (if not present)
- Create a virtual environment at `.venv_cover`
- Install dependencies (TensorFlow, PyTorch, SimplerEnv, LeRobot with PI0, Bridge Verifier, etc.)
- Set up PYTHONPATH for the inference package

**Requirements:** Linux, Python 3.10, CUDA-capable GPU.

### 2. Activate Environment

```bash
source .venv_cover/bin/activate
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

### 3. Run Inference

```bash
cd CoVer_VLA/inference/experiments/robot/simpler/bashes
./test_pi.sh
```

Or run a single task with custom arguments:

```bash
cd CoVer_VLA/inference/experiments/robot/simpler/bashes
python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 100 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8
```

### 4. Visualize Results

After running inference, analyze success rates and generate plots:

```bash
cd CoVer_VLA/inference/experiments/robot/simpler/bashes
python analyze_success_rate.py --output-dir ./analysis_plots
```

Plots are saved to `./analysis_plots/` including:
- Success rates across experiments
- Verifier score distributions over time
- Per-task similarity trajectories
- Evaluation statistics by rollout folder

---

## Project Structure

```
vla-clip/
├── CoVer_VLA/
│   ├── scripts/env_simpler_pi.sh   # Setup script
│   ├── inference/                  # Evaluation and inference
│   │   └── experiments/robot/simpler/
│   │       ├── run_simpler_eval_with_openpi.py
│   │       ├── bashes/
│   │       │   ├── test_pi.sh
│   │       │   └── analyze_success_rate.py
│   │       └── ...
│   └── SimplerEnv/                 # Simulation environment
├── bridge_verifier/                # Action verifier model
├── lerobot_custom/                 # LeRobot with PI0 policy
├── requirements.txt
└── README.md
```

---

## Output Locations

| Output | Path |
|--------|------|
| Logs | `experiments/logs/` (relative to CWD) |
| Rollout videos | `rollouts_openpi_original/` or `rollouts_openpi_rephrase/` |
| Episode data (pickle) | Same as rollout videos |
| Analysis plots | `./analysis_plots/` (or `--output-dir`) |

---

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--task_suite_name` | `simpler_widowx`, `simpler_ood`, `simpler_put_eggplant_in_basket`, etc. |
| `--use_verifier` | Enable/disable action verifier |
| `--policy_batch_inference_size` | Actions sampled per instruction |
| `--lang_rephrase_num` | Number of language rephrases |
| `--num_trials_per_task` | Episodes per task |

---

For more details, see [CoVer_VLA/README.md](CoVer_VLA/README.md).
