"""
run_simpler_api.py

Runs evaluation using the SIMPLER server API instead of local model inference.

Usage:
    # First, start the server:
    # conda activate <env> && \
    # cd ~/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler && \
    # python simpler_server.py
    
    # Then run this script:
    python experiments/robot/simpler/run_simpler_api.py \
        --task_suite_name [ simpler_widowx ... ] \
        --api_url http://localhost:5001 \
        --run_id_note <OPTIONAL TAG> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import itertools
import os
import sys
import json
import pickle
import base64
import requests
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from collections import deque
import draccus
import tqdm
import time
import random

import wandb
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.simpler_utils_robomonkey import (
    get_simpler_dummy_action,
    get_simpler_env,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import imageio

# Define constants locally to avoid importing problematic modules
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

def save_rollout_video_openpi(rollout_images, idx, success, task_description, transformation_type, model_name, lang_rephrase_num, policy_batch_inference_size, log_file=None):
    """Saves an MP4 replay of an episode."""
    if model_name == "juexzz/INTACT-pi0-finetune-rephrase-bridge":
        rollout_dir = f"./rollouts_openpi_rephrase/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    elif model_name == "juexzz/INTACT-pi0-finetune-bridge":
        rollout_dir = f"./rollouts_openpi_original/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

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

def save_episode_data_openpi(episode_data, idx, success, task_description, transformation_type, model_name, lang_rephrase_num, policy_batch_inference_size, log_file=None):
    """Saves episode data (verifier scores, instructions, actions) to a pickle file."""
    if model_name == "juexzz/INTACT-pi0-finetune-rephrase-bridge":
        rollout_dir = f"./rollouts_openpi_rephrase/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    elif model_name == "juexzz/INTACT-pi0-finetune-bridge":
        rollout_dir = f"./rollouts_openpi_original/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    
    pkl_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.pkl"
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(episode_data, f)
    
    print(f"Saved episode data at path {pkl_path}")
    if log_file is not None:
        log_file.write(f"Saved episode data at path {pkl_path}\n")
    return pkl_path

def load_rephrases(task_suite_name: str):
    """Load pre-generated rephrases for the task suite."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'simpler_rephrased_final_eval.json')
    
    with open(json_path, 'r') as f:
        all_rephrases = json.load(f)
    return all_rephrases.get("instructions", {})

def save_image_to_disk(image_array, output_dir="./temp_images"):
    """Save numpy image array to disk and return the path."""
    import os
    import tempfile
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy array to PIL Image
    if image_array.dtype != np.uint8:
        # Normalize to 0-255 if needed
        image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
    
    from PIL import Image
    image = Image.fromarray(image_array)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(dir=output_dir, suffix='.png', delete=False)
    image_path = temp_file.name
    image.save(image_path, format="PNG")
    temp_file.close()
    
    return image_path

def convert_to_serializable(obj):
    """Recursively convert numpy arrays and other non-serializable objects to JSON-serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def call_api_for_action(api_url, instruction, image_array, observation_state, action_history, 
                        rephrased_list=None, lang_rephrase_num=1, policy_batch_inference_size=2,
                        use_verifier=True, action_queue=None, timestep=0, n_action_steps=4, action_ensemble_temp=-0.8,
                        temp_image_dir="./temp_images"):
    """Call the API to get the next action."""
    # Save image to disk and get path
    image_path = save_image_to_disk(image_array, output_dir=temp_image_dir)
    
    # Convert action_queue if it exists (may contain numpy arrays or tensors)
    action_queue_serialized = None
    if action_queue is not None:
        if isinstance(action_queue, list):
            action_queue_serialized = [convert_to_serializable(item) for item in action_queue]
        else:
            action_queue_serialized = convert_to_serializable(action_queue)
    
    payload = {
        'instruction': instruction,
        'image_path': image_path,  # Send path instead of base64
        'observation_state': convert_to_serializable(observation_state),
        'action_history': convert_to_serializable(action_history),
        'rephrased_list': rephrased_list,
        'lang_rephrase_num': lang_rephrase_num,
        'policy_batch_inference_size': policy_batch_inference_size,
        'use_verifier': use_verifier,
        'action_queue': action_queue_serialized,
        'timestep': timestep,
        'n_action_steps': n_action_steps,
        'action_ensemble_temp': action_ensemble_temp,
    }
    
    response = requests.post(
        f"{api_url}/process_action",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
    
    # Clean up temporary image file after server has loaded it
    import os
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return result

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # API Configuration
    #################################################################################################################
    api_url: str = "http://localhost:5001"              # URL of the SIMPLER server API

    #################################################################################################################
    # Model-specific parameters (used for saving rollouts)
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "juexzz/INTACT-pi0-finetune-rephrase-bridge"     # Pretrained checkpoint path (used for directory structure)
    obs_history: int = 1                             # Number of images to pass in from history

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "simpler_widowx"          # Task suite.
    initial_states_type: str = "eval"
    num_steps_wait: int = 0                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 300                    # Number of rollouts per task

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
    use_verifier: bool = True                          # Whether to use the verifier model for similarity scoring
    # Action chunking (match INT-ACT style multi-step actions)
    n_action_steps: int = 4
    
    # Language transformation parameters
    lang_transform_type: str = "no_transform"            # Type of language transformation (rephrase/no_transform)
    lang_rephrase_num: int = 8
    # Batch inference parameters
    policy_batch_inference_size: int = 2                        # Number of samples for batch inference (same instruction repeated)
    
    # Action ensemble parameters (for temporal ensembling)
    action_ensemble_temp: float = -0.8                   # Temperature for action ensembling (negative = more recent actions get more weight)

@draccus.wrap()
def eval_simpler(cfg: GenerateConfig) -> None:
    # Check API connectivity
    health_response = requests.get(f"{cfg.api_url}/health", timeout=5)
    health_response.raise_for_status()
    print(f"Successfully connected to API at {cfg.api_url}")

    # Set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-API-{cfg.task_suite_name}-{DATE_TIME}"
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

    # Initialize SIMPLER task suite
    task_suite = get_benchmark(cfg.task_suite_name)()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    
    # Load pre-generated rephrases if available
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        seeds = itertools.count(1000)

        # Initialize LIBERO environment and task description
        env = get_simpler_env(task, cfg.model_family)
        original_task_description = env.get_language_instruction()
        
        # Use rephrased instruction if available
        if cfg.lang_transform_type == "no_transform":
            task_description = original_task_description
            rephrased_list = None
        else:
            matching_task_id = None
            for task_key, task_data in preloaded_rephrases.items():
                if task_key == original_task_description:
                    matching_task_id = task_key
                    break
            
            if matching_task_id is not None:
                rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"]
                task_description = preloaded_rephrases[matching_task_id]["original"]
                print(f"Using rephrased instruction: {task_description}")
            else:
                raise ValueError(f"No preloaded rephrases found for task: {original_task_description}")

        if cfg.lang_transform_type == "no_transform":
            assert cfg.lang_rephrase_num == 1, "Language rephrase number must be 1 for no transformation"
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for trail_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            if trail_idx % 50 == 0:
                seeds = itertools.count(1000)
            
            # Reset environment to specified seed (initial state)
            obs, reset_info = env.reset(seed=next(seeds))

            # Setup
            t = 0
            replay_images = []
            action_history = []
            action_queue = None
            
            # Track episode data for saving
            episode_data = {
                'verifier_scores': [],
                'selected_instructions': [],
                'execute_actions': [],
                'step_timestamps': [],
                'original_task_description': original_task_description,
                'used_task_description': task_description,
                'success': False,
                'episode_length': 0
            }
            
            if cfg.task_suite_name.startswith("simpler"):
                max_steps = 150
            else:
                raise NotImplementedError(f"Unknown task suite: {cfg.task_suite_name}")

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            pbar = tqdm.tqdm(total=max_steps + cfg.num_steps_wait, desc=f"Episode steps")
            
            max_score = None
            
            while t < max_steps + cfg.num_steps_wait:
                # Wait for objects to stabilize
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(get_simpler_dummy_action(cfg.model_family))
                    t += 1
                    pbar.update(1)
                    continue

                # Get raw image from environment
                raw_img = get_image_from_maniskill2_obs_dict(env, obs)
                replay_images.append(raw_img)

                # Extract observation state - pass full obs dict to match original format
                # The adapter expects obs["agent"]["eef_pos"] with 8 elements (xyz + quaternion xyzw + gripper)
                # So we pass the full obs dict, not just eef_pos
                observation_state = obs
                
                # Print only agent eef_pos for debugging (keep dict structure)
                eef_pos = observation_state.get('agent', {}).get('eef_pos', None)
                obs_server_format = {'agent': {'eef_pos': eef_pos}} if eef_pos is not None else {}
                # print(f"Observation state (eef_pos only): {obs_server_format}")

                # Call API to get action
                api_response = call_api_for_action(
                    api_url=cfg.api_url,
                    instruction=task_description,
                    image_array=raw_img,
                    observation_state=obs_server_format,
                    action_history=action_history,
                    rephrased_list=rephrased_list,
                    lang_rephrase_num=cfg.lang_rephrase_num,
                    policy_batch_inference_size=cfg.policy_batch_inference_size,
                    use_verifier=cfg.use_verifier,
                    action_queue=action_queue,
                    timestep=t,
                    n_action_steps=cfg.n_action_steps,
                    action_ensemble_temp=cfg.action_ensemble_temp,
                    temp_image_dir="./temp_images"  # Directory for temporary images
                )
                
                if api_response.get('status') != 'success':
                    raise ValueError(f"API returned error: {api_response.get('error')}")
                
                # Extract action and metadata
                execute_action = np.array(api_response['action'])
                selected_instruction = api_response.get('selected_instruction', task_description)
                verifier_score = api_response.get('verifier_score')
                action_queue = api_response.get('action_queue')
                action_history_update = api_response.get('action_history_update')
                
                # Update task description if verifier selected a different instruction
                if selected_instruction != task_description:
                    task_description = selected_instruction
                
                # Update action history
                if action_history_update is not None:
                    action_history.append(np.array(action_history_update))
                
                # Store episode data
                episode_data['verifier_scores'].append(verifier_score)
                episode_data['selected_instructions'].append(selected_instruction)
                episode_data['execute_actions'].append(execute_action.copy())
                episode_data['step_timestamps'].append(t)
                
                max_score = verifier_score if verifier_score is not None else max_score

                # Execute action in environment
                obs, reward, done, trunc, info = env.step(execute_action)
                
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                
                t += 1
                
                # Update progress bar
                if cfg.use_verifier and max_score is not None:
                    pbar.set_description(f"Episode steps (score: {max_score:.3f})")
                pbar.update(1)

            pbar.close()
            task_episodes += 1
            total_episodes += 1

            # Update episode data with final information
            episode_data['success'] = done
            episode_data['episode_length'] = t

            # Save a replay video of the episode
            save_rollout_video_openpi(
                replay_images, total_episodes, success=done, 
                task_description=original_task_description, 
                transformation_type=cfg.lang_transform_type,
                model_name=cfg.pretrained_checkpoint,
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=cfg.policy_batch_inference_size,
                log_file=log_file
            )
            
            # Save episode data to pickle file
            save_episode_data_openpi(
                episode_data, total_episodes, success=done, 
                task_description=original_task_description, 
                transformation_type=cfg.lang_transform_type,
                model_name=cfg.pretrained_checkpoint,
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=cfg.policy_batch_inference_size,
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

