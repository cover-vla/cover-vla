# SimplerEnv Evaluation with Batch Verifier and VLA-CLIP Trajectory Scoring

This directory contains the enhanced evaluation script `run_simpler_eval_with_verifier.py` that combines:

1. **SimplerEnv environment** for robot manipulation tasks
2. **Batch language verifier** using SGLang for efficient multi-instruction processing
3. **VLA-CLIP trajectory scoring** for action verification and selection

## Overview

The new evaluation pipeline works as follows:

1. **Task Setup**: Load a SimplerEnv task (e.g., stacking cubes, placing objects)
2. **Language Generation**: Generate multiple rephrased versions of the task instruction
3. **Batch Action Generation**: Use the SGLang batch server to generate actions for all instructions simultaneously
4. **Action Scoring**: Score each action using the VLA-CLIP trajectory verifier
5. **Action Selection**: Choose the best action based on scores and strategy
6. **Execution**: Execute the selected action in the environment
7. **Repeat**: Continue until task completion or timeout

## Prerequisites

### 1. SGLang Batch Server
Start the SGLang batch server from the `sglang-batch-lang` directory:

```bash
cd sglang-batch-lang
python vla/openvla_server.py --seed 0
```

The server will be available at `http://localhost:3200`.

### 2. VLA-CLIP Trajectory Model
You need a trained VLA-CLIP trajectory model for action scoring. This should be trained using the `bridge_verifier` pipeline.

### 3. Environment Setup
Ensure you have all the required dependencies installed:

```bash
# Install json_numpy for batch server communication
pip install json_numpy

# Install requests for HTTP communication
pip install requests

# Ensure clip_verifier scripts are available
# The script will automatically add the path to sys.path
```

## Usage

### Basic Usage (Batch Verifier Only)

```bash
python experiments/robot/simpler/run_simpler_eval_with_verifier.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/vla/checkpoint \
    --task_suite_name simpler_widowx \
    --use_batch_verifier True \
    --batch_server_url http://localhost:3200 \
    --clip_select_action_num_candidates 3 \
    --num_trials_per_task 10
```

### Full Usage (With VLA-CLIP Trajectory Scoring)

```bash
python experiments/robot/simpler/run_simpler_eval_with_verifier.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/vla/checkpoint \
    --task_suite_name simpler_widowx \
    --use_batch_verifier True \
    --batch_server_url http://localhost:3200 \
    --batch_temperature 1.0 \
    --use_vla_clip_trajectory_scorer True \
    --vla_clip_traj_model_path /path/to/trajectory/model.pt \
    --vla_clip_history_length 10 \
    --vla_clip_use_transformer True \
    --clip_select_action_num_candidates 3 \
    --clip_select_action_strategy highest_score \
    --vla_clip_score_threshold 0.5 \
    --lang_transform_type rephrase \
    --num_trials_per_task 20 \
    --use_wandb True \
    --wandb_project your_project \
    --wandb_entity your_entity
```

## Configuration Parameters

### Model Parameters
- `--model_family`: Model family (openvla, prismatic)
- `--pretrained_checkpoint`: Path to VLA model checkpoint
- `--center_crop`: Whether to center crop images (True if model trained with augmentations)

### Batch Verifier Parameters
- `--use_batch_verifier`: Enable batch language verifier (True/False)
- `--batch_server_url`: URL of SGLang batch server
- `--batch_temperature`: Temperature for batch inference sampling

### VLA-CLIP Trajectory Scorer Parameters
- `--use_vla_clip_trajectory_scorer`: Enable trajectory scoring (True/False)
- `--vla_clip_traj_model_path`: Path to trained trajectory model
- `--vla_clip_history_length`: Action history length (must match training)
- `--vla_clip_use_transformer`: Use transformer for action encoding (True/False)
- `--clip_select_action_num_candidates`: Number of candidate instructions to evaluate
- `--clip_select_action_strategy`: Action selection strategy (highest_score/softmax_sample)
- `--vla_clip_score_threshold`: Score threshold for triggering alternative action selection

### Language Transformation Parameters
- `--lang_transform_type`: Type of language transformation (rephrase/no_transform)
- `--use_original_task_description`: Use original task description for scoring (True/False)

### Environment Parameters
- `--task_suite_name`: Task suite name (simpler_widowx, simpler_stack_cube, etc.)
- `--num_trials_per_task`: Number of evaluation trials per task
- `--num_steps_wait`: Steps to wait for environment stabilization

### Logging Parameters
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity name
- `--run_id_note`: Optional note for run identification

## Task Suites

Available task suites:
- `simpler_widowx`: All WidowX tasks (stack cube, put eggplant in basket, carrot on plate, spoon on towel)
- `simpler_stack_cube`: Only cube stacking task
- `simpler_put_eggplant_in_basket`: Only eggplant in basket task
- `simpler_carrot_on_plate`: Only carrot on plate task
- `simpler_spoon_on_towel`: Only spoon on towel task

## Language Rephrases

The script uses pre-generated rephrases stored in `libero_rephrase_pos_rephrase_neg_negation.json`. Each task has multiple rephrased versions of the original instruction to provide diversity in action generation.

If the rephrase file is not found, the script will fall back to on-the-fly generation using the `LangTransform` class.

## Output

The script generates:
1. **Local logs**: Detailed evaluation logs saved to `./experiments/logs/`
2. **Videos**: Rollout videos for successful and failed episodes
3. **W&B logs**: Metrics and videos if W&B is enabled
4. **Console output**: Real-time progress and results

## Troubleshooting

### Batch Server Connection Issues
- Ensure the SGLang batch server is running on the specified URL
- Check that the server is accessible and responding to requests
- Verify the image path format is compatible with the server

### VLA-CLIP Model Issues
- Ensure the trajectory model path is correct
- Verify the history length matches the training configuration
- Check that the model uses the same action dimension as the environment

### Memory Issues
- Reduce batch size or number of candidates if running out of memory
- Use gradient checkpointing if available
- Consider using CPU for trajectory scoring if GPU memory is limited

## Example Results

The script will output results like:
```
Task 0 ('stack the cube on top of the other cube') Success Rate: 0.80 (8/10)
Task 1 ('put the eggplant in the basket') Success Rate: 0.70 (7/10)
...
Overall Success Rate: 0.750 (15/20)
```

With detailed logging of action selection decisions and VLA-CLIP scores for each step. 