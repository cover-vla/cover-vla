# VLA-CLIP

**Vision-Language-Action Models with Instruction Verification for Robot Manipulation**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Website](https://img.shields.io/badge/Project-Website-blue?style=for-the-badge)](#)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/stanfordasl/CoVer-BridgeV2)
[![License](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](LICENSE)

## 📋 To-Do

- [x] Initial release to Bridge env *(done)*
- [ ] Develop CoVer verifier server
- [ ] Update DROID evaluation script with PolaRis

## Table of contents
- [To-Do](#to-do)
- [Setup](#setup)
- [Action Verifier](#action-verifier)
- [SIMPLER Environment](#simpler-environment)
- [Evaluation Results](#evaluation-results)
- [Acknowledgements](#acknowledgements)
- [Troubleshooting](#troubleshooting)

## 🛠️ Setup

Clone this repository:

```bash
git clone --recurse-submodules <REPO_URL>
```

Use the provided script to set up all dependencies:

```bash
bash CoVer_VLA/scripts/env_simpler_pi.sh
```

This script will:
- Install [uv](https://github.com/astral-sh/uv) (if not present)
- Create a virtual environment at `.venv_cover`
- Install dependencies (TensorFlow, PyTorch, SimplerEnv, LeRobot with PI0, Bridge Verifier, etc.)
- Set up PYTHONPATH for the inference package

**Requirements:** Linux, Python 3.10, CUDA-capable GPU.

## ✅ Action Verifier

Download the pretrained checkpoint and spin up the action verifier:

```bash
cd bridge_verifier
huggingface-cli download stanfordasl/CoVer-BridgeV2 cover_verifier_bridge.pt --local-dir .
# Or: hf download stanfordasl/CoVer-BridgeV2 cover_verifier_bridge.pt --local-dir .
cd ..
```

The checkpoint (~312MB) will be saved to `bridge_verifier/cover_verifier_bridge.pt`.

## 🤖 SIMPLER Environment

### Running VLA-CLIP

Activate the environment and run the evaluation script as follows:

```bash
source .venv_cover/bin/activate
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

cd CoVer_VLA/inference/experiments/robot/simpler/bashes
./test_pi.sh
```

Or run a single task with custom arguments:

```bash
python ../run_simpler_eval_with_openpi.py \
    --task_suite_name simpler_widowx \
    --lang_transform_type rephrase \
    --pretrained_checkpoint juexzz/INTACT-pi0-finetune-bridge \
    --num_trials_per_task 100 \
    --use_verifier True \
    --policy_batch_inference_size 5 \
    --lang_rephrase_num 8
```

- `policy_batch_inference_size`: Number of actions sampled per instruction.
- `lang_rephrase_num`: Number of language rephrases.
- `task_suite_name`: simpler_widowx, simpler_ood, simpler_put_eggplant_in_basket, etc.

### Baseline without Verifier

To disable the verifier and use the base policy only:

```bash
--use_verifier False
```

### Visualize Results

After running inference, analyze success rates and generate plots:

```bash
python analyze_success_rate.py --output-dir ./analysis_plots
```

## 📊 Evaluation Results

| Task | Policy Batch | Lang Rephrases | Seed 1 | Seed 2 | Seed 3 | Average | Baseline | Success Rate ↑ |
|------|--------------|----------------|--------|--------|--------|---------|----------|----------------|
| Task A | - | - | - | - | - | - | - | - |
| Task B | - | - | - | - | - | - | - | - |

Logs are saved under: `experiments/logs/` (relative to CWD). Rollout videos: `rollouts_openpi_original/` or `rollouts_openpi_rephrase/`.

## 📚 Acknowledgements

We thank the authors of [LeRobot](https://github.com/huggingface/lerobot), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [CoVer](https://github.com/stanfordasl/CoVer), and related projects for their contributions to the open-source community. Our implementation builds upon these projects.

If you find this project helpful, please consider citing:

```bibtex
@article{vla-clip2025,
  title={VLA-CLIP: Vision-Language-Action Models with Instruction Verification for Robot Manipulation},
  author={Author names},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
}
```

## 🔎 Troubleshooting

**MuJoCo / OpenGL rendering:** If you encounter display or rendering issues, ensure:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

**Vulkan error:** If you see `No Vulkan extensions found for window surface creation`, you may need to install Vulkan dependencies or use `osmesa` as above.

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

For more details, see [CoVer_VLA/README.md](CoVer_VLA/README.md).
