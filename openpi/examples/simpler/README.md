# SimplerEnv Integration for OpenPI

This directory contains the integration of SimplerEnv with OpenPI, enabling evaluation of OpenPI models on SimplerEnv tasks. This integration mirrors the existing LIBERO integration pattern.

## Overview

SimplerEnv is a simulated manipulation environment used for evaluating robot policies. This integration allows you to:

- Evaluate OpenPI models (π₀, π₀-FAST, π₀.₅) on SimplerEnv tasks
- Run evaluations using the same task suites as RoboMonkey
- Save trajectories and videos of policy rollouts
- Use both individual tasks and grouped task suites

## Task Suites

### Main Task Suites (from RoboMonkey)

- **`simpler_widowx`**: Main WidowX tasks (4 tasks)
  - `widowx_stack_cube`, `widowx_put_eggplant_in_basket`, `widowx_carrot_on_plate`, `widowx_spoon_on_towel`

- **`simpler_ood`**: Out-of-distribution tasks (2 tasks)
  - `widowx_redbull_on_plate`, `widowx_zucchini_on_towel`

### Individual Task Suites

- `simpler_stack_cube`: Only cube stacking
- `simpler_put_eggplant_in_basket`: Only eggplant in basket
- `simpler_spoon_on_towel`: Only spoon on towel
- `simpler_carrot_on_plate`: Only carrot on plate
- `simpler_redbull_on_plate`: Only RedBull on plate
- `simpler_carrot_on_plate_unseen_lighting`: Carrot with unseen lighting
- `simpler_tennis_ball_in_basket`: Only tennis ball in basket
- `simpler_toy_dinosaur_on_towel`: Only toy dinosaur on towel
- `simpler_zucchini_on_towel`: Only zucchini on towel

### Google Robot Tasks (from SimplerEnv)

- **`google_robot_basic`**: Basic Google Robot tasks
  - `google_robot_pick_coke_can`, `google_robot_move_near`, `google_robot_open_drawer`, `google_robot_close_drawer`

- **`google_robot_pick`**: All picking tasks
  - `google_robot_pick_coke_can`, `google_robot_pick_horizontal_coke_can`, `google_robot_pick_vertical_coke_can`, `google_robot_pick_standing_coke_can`, `google_robot_pick_object`

- **`google_robot_drawer`**: All drawer tasks
  - `google_robot_open_drawer`, `google_robot_open_top_drawer`, `google_robot_open_middle_drawer`, `google_robot_open_bottom_drawer`, `google_robot_close_drawer`, `google_robot_close_top_drawer`, `google_robot_close_middle_drawer`, `google_robot_close_bottom_drawer`

- **`google_robot_place`**: All placement tasks
  - `google_robot_place_in_closed_drawer`, `google_robot_place_in_closed_top_drawer`, `google_robot_place_in_closed_middle_drawer`, `google_robot_place_in_closed_bottom_drawer`, `google_robot_place_apple_in_closed_top_drawer`

## Local Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- SimplerEnv installed
- OpenPI dependencies

### Installation

1. **Create virtual environment:**
   ```bash
   cd openpi/examples/simpler
   uv venv --python 3.10 .venv
   source .venv/bin/activate
   ```

2. **Install ManiSkill2 dependencies:**
   ```bash
   uv pip install -r ../../../RoboMonkey/SimplerEnv/ManiSkill2_real2sim/requirements.txt
   ```

3. **Install SimplerEnv:**
   ```bash
   uv pip install -e ../../../RoboMonkey/SimplerEnv/ManiSkill2_real2sim
   uv pip install -e ../../../RoboMonkey/SimplerEnv
   ```

4. **Install OpenPI client:**
   ```bash
   uv pip install -e ../../packages/openpi-client
   ```

5. **Install additional packages:**
   ```bash
   uv pip install tyro
   ```

### Running Evaluation

1. **Start the policy server:**
   ```bash
   # From the openpi root directory
   uv run scripts/serve_policy.py --env LIBERO  # Use LIBERO mode for SimplerEnv
   ```

2. **Run evaluation (in another terminal):**
   ```bash
   cd examples/simpler
   source .venv/bin/activate
   python main.py --task_suite_name simpler_widowx --num_trials_per_task 5
   ```

## Configuration Options

### Command Line Arguments

- `--host`: Policy server host (default: "0.0.0.0")
- `--port`: Policy server port (default: 8000)
- `--task_suite_name`: Task suite to evaluate (default: "simpler_widowx")
- `--num_trials_per_task`: Number of rollouts per task (default: 20)
- `--num_steps_wait`: Steps to wait for environment stabilization (default: 10)
- `--save_trajectories`: Save trajectory data (default: False)
- `--recording`: Save video recordings (default: False)
- `--video_dir`: Directory for videos (default: "videos")
- `--trajectory_dir`: Directory for trajectories (default: "trajectories")
- `--resize_size`: Image resize size (default: 224)
- `--replan_steps`: Action chunk size (default: 5)

### Example Commands

```bash
# Evaluate main WidowX tasks
python main.py --task_suite_name simpler_widowx --num_trials_per_task 10

# Evaluate out-of-distribution tasks
python main.py --task_suite_name simpler_ood --recording

# Evaluate single task with video recording
python main.py --task_suite_name simpler_stack_cube --recording --save_trajectories

# Evaluate Google Robot tasks
python main.py --task_suite_name google_robot_basic --num_trials_per_task 5
```

## Output

The evaluation will produce:

- **Console output**: Success rates and progress information
- **Videos** (if `--recording` enabled): MP4 files in `videos/` directory
- **Trajectories** (if `--save_trajectories` enabled): Pickle files in `trajectories/` directory

## Integration Details

This integration follows the same pattern as the existing LIBERO integration:

1. **Policy Classes**: `SimplerInputs` and `SimplerOutputs` in `src/openpi/policies/simpler_policy.py`
2. **Training Config**: Added SimplerEnv configurations in `src/openpi/training/config.py`
3. **Evaluation Script**: `main.py` handles environment interaction and policy evaluation
4. **Data Format**: Compatible with LeRobot dataset format for training/fine-tuning

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure CUDA is properly installed
2. **Environment errors**: Check that SimplerEnv is properly installed
3. **Policy server connection**: Verify the policy server is running and accessible
4. **Task not found**: Ensure the task name exists in SimplerEnv

### Debug Mode

Add `--verbose` flag for detailed logging:
```bash
python main.py --task_suite_name simpler_widowx --verbose
```