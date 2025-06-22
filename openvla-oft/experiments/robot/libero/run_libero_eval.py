"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import copy
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import torch

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

sys.path.append("/home/xilun/vla-clip/clip_verifier/scripts")
from vla_clip_inference import VLA_CLIP_Inference
from vla_dino_inference import VLA_DINO_Inference
from lang_transform import LangTransform

torch.set_num_threads(8)

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    use_vla_clip_trajectory_scorer: bool = False           # Enable the trajectory scorer?
    use_vla_dino_trajectory_scorer: bool = False          # Enable the trajectory scorer?
    vla_clip_traj_model_path: Optional[str] = None         # Path to the trajectory VLA-CLIP model
    vla_clip_history_length: int = 10                      # History length (MUST match model training)
    clip_select_action_num_candidates: int = 3             # Number of candidate instructions (incl. original) for action selection
    clip_select_action_strategy: str = "highest_score"     # Strategy: 'highest_score' or 'softmax_sample'
    vla_clip_score_threshold: float = 0.3                # Threshold to trigger candidate generation/evaluation

    lang_transform_type: str = "rephrase" 
    use_original_task_description: bool = False
    
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.eval()
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action

def load_rephrases(json_path, suite_name):
    with open(json_path, 'r') as f:
        all_rephrases = json.load(f)
    return all_rephrases[suite_name]


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    rephrased_list=None,
    vla_scorer=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    all_scores = []  # Track CLIP scores for each step
    all_actions = [] # Track executed actions for each step
    
    if cfg.clip_select_action_num_candidates > 1:
        # pre_sampled_all_language_instructions = lang_transform.transform(task_description,cfg.lang_transform_type, batch_number=10)
        pre_sampled_all_language_instructions = rephrased_list[1:]  # Use the rest as alternatives
    
    assert cfg.use_vla_clip_trajectory_scorer or cfg.use_vla_dino_trajectory_scorer, "Must use at least one trajectory scorer!"
    assert not (cfg.use_vla_clip_trajectory_scorer and cfg.use_vla_dino_trajectory_scorer), "Cannot use both trajectory scorers!"
    
    # Run episode
    success = False 
    best_instr = task_description
    while t < max_steps + cfg.num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)
        image_for_clip_tuple = (observation["full_image"], observation["wrist_image"])
        # Always generate a new action chunk at every step
        actions = get_action(
            cfg,
            model,
            copy.deepcopy(observation),
            best_instr,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )
        # --- CLIP verifier and rejection sampling logic ---
        step_score = None
        if cfg.use_vla_clip_trajectory_scorer or cfg.use_vla_dino_trajectory_scorer:
            # Score the entire action chunk
            original_score = vla_scorer.get_history_score(
                image_for_clip_tuple,
                best_instr,
                process_action(np.array(actions.copy()), cfg.model_family)  # shape (chunk_size, action_dim)
            ).detach().cpu().numpy().squeeze()
            best_actions = actions.copy()
            best_score = original_score
            # If score is below threshold and we want alternatives
            if (
                cfg.clip_select_action_num_candidates > 1 and
                not np.isnan(original_score) and
                original_score < cfg.vla_clip_score_threshold
            ):
                candidate_instructions = [task_description]
                predicted_action_chunks = [actions]
                scores = [original_score]
                # Sample additional language instructions
                num_to_generate = cfg.clip_select_action_num_candidates - 1
                if num_to_generate > 0 and len(pre_sampled_all_language_instructions) > 0:
                    # sample_indices = np.random.choice(len(pre_sampled_all_language_instructions), size=num_to_generate, replace=False)
                    sample_indices = np.arange(len(pre_sampled_all_language_instructions))[:num_to_generate]
                    additional_instructions = [pre_sampled_all_language_instructions[i] for i in sample_indices]
                    candidate_instructions.extend(additional_instructions)
                    for instr in additional_instructions:
                        a_chunk = get_action(
                            cfg,
                            model,
                            copy.deepcopy(observation),
                            instr,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio_projector,
                            noisy_action_projector=noisy_action_projector,
                            use_film=cfg.use_film,
                        )
                        predicted_action_chunks.append(a_chunk)
                    # Batch score all candidates
                    predicted_action, scores_dict = vla_scorer.predict(
                        image_for_clip_tuple,
                        candidate_instructions,  # Pass the list of instructions, not just one
                        [process_action(np.array(a_chunk), cfg.model_family) for a_chunk in predicted_action_chunks]
                    )
                    scores = [scores_dict[str(i)] for i in range(len(predicted_action_chunks))]
                # Select the best action chunk
                scores = np.array(scores)
                # print all the scores with corresponding insturction and action 
                # for i in range(len(candidate_instructions)):
                #     print(f"Instruction: {candidate_instructions[i]}, Action: {predicted_action_chunks[i][0]}, Score: {scores[i]}")
                # input()
                valid_indices = np.where(scores > -np.inf)[0]
                if len(valid_indices) > 0:
                    best_idx = valid_indices[np.argmax(scores[valid_indices])]
                    best_actions = predicted_action_chunks[best_idx]
                    best_score = scores[best_idx]
                    best_instr = candidate_instructions[best_idx]
                    if best_idx != 0:
                        print(f"[t={t}] Selected alternative action chunk via: '{best_instr}' (Score: {best_score:.3f})")
                    else:
                        print(f"[t={t}] Kept original action chunk (Score: {best_score:.3f}) after evaluating alternatives.")
                else:
                    print("Warning: All candidate scores are invalid (-inf). Using original action chunk.")
            actions = best_actions
            step_score = best_score
        else:
            step_score = None
        # Always execute the first action in the chosen chunk
        action = actions[0]
        action = process_action(action, cfg.model_family)
        # Save score and action for logging
        all_scores.append(step_score)
        all_actions.append(action.tolist())
        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        print(f"t={t}, action={action.tolist()}, reward={step_score}")
        t += 1


    return success, replay_images, all_scores, all_actions


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    rephrased_list=None,
    vla_scorer=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, original_task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    if cfg.lang_transform_type == "no_transform":
        task_description = original_task_description
    else:
        task_description = rephrased_list[0]  # Use the first as the main instruction
        # task_description = original_task_description
    
    rephrased_list[3] = original_task_description

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        

        # Run episode
        success, replay_images, all_scores, all_actions = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            rephrased_list,
            vla_scorer,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, transform_type=cfg.lang_transform_type, log_file=log_file,
            score_list=all_scores, action_list=all_actions, clip_update_num=cfg.clip_select_action_num_candidates)

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    if cfg.use_vla_clip_trajectory_scorer:
        vla_scorer = VLA_CLIP_Inference(
            model_path=cfg.vla_clip_traj_model_path,
            history_length=cfg.vla_clip_history_length,
            use_transformer=True
        )
    elif cfg.use_vla_dino_trajectory_scorer:
        vla_scorer = VLA_DINO_Inference(
            model_path=cfg.vla_clip_traj_model_path,
            history_length=cfg.vla_clip_history_length,
            use_transformer=True
        )
    else:
        raise ValueError("Must use at least one trajectory scorer!")
    lang_transform = LangTransform()
    
    # Load pre-generated rephrases if available
    rephrases_json_path = f"/home/xilun/vla-clip/openvla/experiments/robot/libero/libero_rephrases.json"
    preloaded_rephrases = load_rephrases(rephrases_json_path, cfg.task_suite_name)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)[5:]):
        
        # --- Always use the first rephrase as the main instruction, rest as alternatives ---
        rephrased_list = preloaded_rephrases[str(task_id)]["rephrases"]
        # if task_id > 0:
        #     break
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            rephrased_list,
            vla_scorer,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
