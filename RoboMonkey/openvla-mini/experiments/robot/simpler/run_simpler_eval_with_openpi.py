"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/simpler/run_simpler_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ simpler_widowx ... ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import itertools
import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm

import wandb
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.simpler_utils_robomonkey import (
    convert_maniskill_with_bridge_adapter,
    get_simpler_dummy_action,
    get_simpler_env,
    get_simpler_img,
    create_bridge_adapter_wrapper,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
import time
import random

# Define constants locally to avoid importing problematic modules
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Removed denormalization function - PI0 model already unnormalizes actions internally

def get_image_resize_size(cfg):
    """Get image resize size based on model family."""
    return 224

def set_seed_everywhere(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_rollout_video_openpi(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    import imageio
    rollout_dir = f"./rollouts_clip/openpi"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

# import sys
sys.path.append('/root/vla-clip/lerobot_intact')
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import torch

def load_rephrases(task_suite_name: str):
    """Load pre-generated rephrases for the task suite."""
    # Make the path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'simpler_rephrased_final_eval.json')
    
    with open(json_path, 'r') as f:
        all_rephrases = json.load(f)
    # The new format has an "instructions" key containing the task data
    return all_rephrases.get("instructions", {})


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    # pretrained_checkpoint: Union[str, Path] = "juexzz/INTACT-pi0-finetune-bridge"     # Pretrained checkpoint path
    pretrained_checkpoint: Union[str, Path] = "juexzz/INTACT-pi0-finetune-rephrase-bridge"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "simpler_widowx"          # Task suite.
    initial_states_type: str = "eval"
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 0                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 30                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on

    # Robomonkey Config
    initial_samples: int = 5
    augmented_samples: int = 32
    # Action chunking (match INT-ACT style multi-step actions)
    n_action_steps: int = 4
    
    # Language transformation parameters
    lang_transform_type: str = "no_transform"            # Type of language transformation (rephrase/no_transform)
    
    # Batch inference parameters
    batch_inference_size: int = 1                        # Number of samples for batch inference (same instruction repeated)
    
    # Action ensemble parameters (for temporal ensembling)
    action_ensemble_temp: float = -0.8                   # Temperature for action ensembling (negative = more recent actions get more weight)

@draccus.wrap()
def eval_simpler(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    assert cfg.initial_samples > 0, "Invalid initial_samples: should be > 0"
    assert cfg.augmented_samples > 0, "Invalid augmented_samples: should be > 0"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    else:
        cfg.unnorm_key = "bridge_orig"

    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )


    # Initialize PI0 policy using lerobot_intact version
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    
    # Use the lerobot_intact version which has PaliGemmaWithExpert support
    pi0_policy = PI0Policy.from_pretrained(cfg.pretrained_checkpoint)
    # Ensure model is on GPU if available (override config device accordingly)
    if torch.cuda.is_available():
        pi0_policy.to("cuda")
        pi0_policy.config.device = "cuda"
    # Apply configured number of action steps (multi-step chunking)
    pi0_policy.config.n_action_steps = int(cfg.n_action_steps)
    print(f"PI0Policy device: {pi0_policy.config.device}")
    
    # Initialize SIMPLER task suite

    task_suite = get_benchmark(cfg.task_suite_name)()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    
    # Load pre-generated rephrases if available
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)
    
    # Create adapter for preprocessing (singleton pattern)
    if not hasattr(pi0_policy, '_preprocess_adapter'):
        pi0_policy._preprocess_adapter = create_bridge_adapter_wrapper(cfg.action_ensemble_temp)
    preprocess_adapter = pi0_policy._preprocess_adapter

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default SIMPLER envs
        if cfg.initial_states_type == "eval":
            seeds = itertools.count(1000)
        elif cfg.initial_states_type == "train":
            seeds = itertools.count(0)
        else:
            raise ValueError("Unsupported initial states type")

        # Initialize LIBERO environment and task description
        env = get_simpler_env(task, cfg.model_family)
        original_task_description = env.get_language_instruction()
        
        # Use rephrased instruction if available
        if cfg.lang_transform_type == "no_transform":
            task_description = original_task_description
        else:
            # Find matching task in preloaded rephrases
            matching_task_id = None
            for task_key, task_data in preloaded_rephrases.items():
                if task_key == original_task_description:
                    matching_task_id = task_key
                    break
            
            if matching_task_id is not None:
                rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"]
                # Use the first rephrase (like setting number of samples to 1)
                task_description = preloaded_rephrases[matching_task_id]["original"]
                print(f"Using rephrased instruction: {task_description}")
            else:
                print(f"No preloaded rephrases found for task: {original_task_description}, using original")
                task_description = original_task_description

        # Start episodes
        task_episodes, task_successes = 0, 0
        for _ in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment to specified seed (initial state)
            obs, reset_info = env.reset(seed=next(seeds))

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            elif cfg.task_suite_name.startswith("simpler"):
                max_steps = 150  # data is at 5Hz, so this is 30 seconds
            else:
                raise NotImplementedError

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            # Create progress bar for each episode
            pbar = tqdm.tqdm(total=max_steps + cfg.num_steps_wait, desc=f"Episode steps")
            while t < max_steps + cfg.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(get_simpler_dummy_action(cfg.model_family))
                    t += 1
                    pbar.update(1)
                    continue

                # Get raw image from environment (let BridgeSimplerAdapter handle preprocessing)
                # This matches INT-ACT's approach: pass raw images to adapter for consistent preprocessing
                raw_img = get_image_from_maniskill2_obs_dict(env, obs)
                
                replay_images.append(raw_img)

                # buffering #obs_history images, optionally
                image_history = replay_images[-cfg.obs_history :]
                if len(image_history) < cfg.obs_history:
                    image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))
                
                # Prepare observations in the format expected by BridgeSimplerAdapter
                # INT-ACT passes raw observation directly - obs["agent"]["eef_pos"] already has correct format!
                obs_for_adapter = {
                    'observation.images.top': raw_img,  # Raw image from environment
                    'observation.state': obs,  # Raw observation state (matches INT-ACT exactly)
                    'task': task_description
                }
                
                # Use BridgeSimplerAdapter preprocessing
                processed_obs = preprocess_adapter.preprocess(obs_for_adapter)
                
                # Move to policy device
                policy_device = torch.device(pi0_policy.config.device)
                processed_obs = {
                    k: (v.to(device=policy_device) if isinstance(v, torch.Tensor) else v)
                    for k, v in processed_obs.items()
                }
                
                # Determine expected image feature key from the policy config
                image_feature_keys = list(pi0_policy.config.image_features.keys())
                image_key = image_feature_keys[0]
                
                # Create batch of language instructions (same instruction repeated for batch inference)
                batch_size = cfg.batch_inference_size
                # processed_obs['task'] is already a list from BridgeSimplerAdapter
                single_task = processed_obs['task']  # Already a list
                # Create list of identical task lists by extending the original list
                batch_task_instructions = []
                for _ in range(batch_size):
                    batch_task_instructions.extend(single_task)
                # Create batch observation dict for LeRobot PI0
                # Replicate image and state to match batch size
                batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
                batch_state = processed_obs['observation.state'].repeat(batch_size, 1)
                
                observation = {
                    image_key: batch_image,  # [batch_size, 3, 224, 224]
                    "observation.state": batch_state,  # [batch_size, 7]
                    "task": batch_task_instructions,  # List of 5 identical instructions
                }
                
                with torch.no_grad():
                    action_queue = pi0_policy.select_action(observation)
                    # select_action returns a deque; get the first batch of actions
                    action = action_queue.popleft().cpu().numpy()  # Shape: [batch_size, action_dim]
                    
                # print ("action shape:", action.shape)
                # # DEBUG: Check batch inference results
                # print(f"\n=== BATCH INFERENCE (batch_size={batch_size}, temp={cfg.action_ensemble_temp}) ===")
                # print(f"Task instruction: {single_task}")
                # print(f"PI0 raw action shape: {action.shape}")
                # print(f"PI0 raw action (first sample): {action[0]}")
                # if batch_size > 1:
                #     print(f"PI0 raw action (all samples):")
                #     for i in range(batch_size):
                #         print(f"  Sample {i+1}: {action[i]}")
                # print("=== END BATCH INFERENCE ===\n")
                
                # Process action using BridgeSimplerAdapter
                # Use the first action from the batch for environment execution
                action_for_env = action[0:1]  # Keep batch dimension for adapter
                processed_action = convert_maniskill_with_bridge_adapter(action_for_env, cfg.action_ensemble_temp)
                
                # Execute action in environment
                obs, reward, done, trunc, info = env.step(processed_action)
                # print("Executed action:", processed_action)
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                pbar.update(1)


            pbar.close()
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video_openpi(
                replay_images, total_episodes, success=done, 
                task_description=original_task_description, 
                log_file=log_file
            )

            # Save at most 5 successes and at most 5 failures
            if cfg.use_wandb and ((done and task_successes < 5) or (not done and task_episodes - task_successes < 5)):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log(
                    {f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))}
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_simpler()
