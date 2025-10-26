"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

This variant delegates action inference to the external PI0 API server
instead of performing local inference. All other evaluation logic remains
the same (task loops, video/log saving, etc.).

Usage:
    # Ensure the PI0 API server is running (port 5001 by default):
    #   conda activate sglang
    #   cd ~/vla-api/lab_infer
    #   python pi0_server.py

    # Then run evaluation here:
    python experiments/robot/simpler/run_simpler_eval_with_openpi_server.ppy \
        --task_suite_name simpler_widowx \
        --run_id_note api
"""

import itertools
import os
import sys
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import base64
import io
import requests
import draccus
import numpy as np
import tqdm

import wandb
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import imageio
from PIL import Image

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
import time
import random

# Define constants locally to avoid importing problematic modules
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_image_resize_size(cfg):
    """Get image resize size based on model family."""
    return 224


def set_seed_everywhere(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_rollout_video_openpi(rollout_images, idx, success, task_description, transformation_type, model_name, lang_rephrase_num, policy_batch_inference_size, log_file=None):
    """Saves an MP4 replay of an episode."""
    if model_name == "juexzz/INTACT-pi0-finetune-rephrase-bridge":
        rollout_dir = f"./rollouts_openpi_rephrase/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    elif model_name == "juexzz/INTACT-pi0-finetune-bridge":
        rollout_dir = f"./rollouts_openpi_original/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    else:
        rollout_dir = f"./rollouts_openpi_api/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"

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
        rollout_dir = f"./rollouts_openpi_api/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"

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


def encode_image_to_base64(image_np: np.ndarray) -> str:
    """Encode HxWxC uint8 image to base64 JPEG string."""
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)
    image = Image.fromarray(image_np)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model/API-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family (kept for parity)
    hf_token: str = Path(".hf_token")               # Unused here; passed server-side
    pretrained_checkpoint: Union[str, Path] = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

    # Remote servers
    pi0_api_url: str = "http://localhost:5001"      # PI0 API base URL

    center_crop: bool = True                         # Parity with original
    obs_history: int = 1                             # Number of images to pass in from history

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "simpler_widowx"          # Task suite.
    initial_states_type: str = "eval"
    num_steps_wait: int = 0
    num_trials_per_task: int = 300

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    prefix: str = ''

    use_wandb: bool = False
    wandb_project: str = "prismatic"
    wandb_entity: Optional[str] = None

    seed: int = 7                                    # Random Seed (for reproducibility)

    # Server-driven inference controls (mirrors original config fields)
    n_action_steps: int = 4
    lang_transform_type: str = "no_transform"
    lang_rephrase_num: int = 1                       # default 1 when no transform
    policy_batch_inference_size: int = 2

    # fmt: on


def initialize_pi0_remote(pi0_api_url: str, checkpoint: Union[str, Path]):
    """Initialize remote PI0 model via API."""
    try:
        resp = requests.post(f"{pi0_api_url}/initialize", json={"pi0_checkpoint": str(checkpoint)}, timeout=120)
        resp.raise_for_status()
        print(f"Initialized remote PI0 model: {resp.json()}")
    except Exception as e:
        print(f"Warning: failed to initialize remote PI0 model at {pi0_api_url}: {e}")


def request_actions_from_server(pi0_api_url: str, instruction: str, image_np: np.ndarray, state_vec: Union[list, np.ndarray], policy_batch_inference_size: int, lang_rephrase_num: int):
    """Call PI0 API to get actions for a single timestep."""
    image_b64 = encode_image_to_base64(image_np)
    if isinstance(state_vec, np.ndarray):
        state_payload = state_vec.tolist()
    else:
        state_payload = state_vec
    payload = {
        "instruction": instruction,
        "image": image_b64,
        "state": state_payload,
        "policy_batch_inference_size": int(policy_batch_inference_size),
        "lang_rephrase_num": int(lang_rephrase_num),
    }
    resp = requests.post(f"{pi0_api_url}/process_image", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


@draccus.wrap()
def eval_simpler(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    # Enforce no-rephrase count when no_transform
    if cfg.lang_transform_type == "no_transform":
        assert cfg.lang_rephrase_num == 1, "Language rephrase number must be 1 for no transformation"

    # Set random seed
    set_seed_everywhere(cfg.seed)

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

    # Initialize remote PI0 server with the desired checkpoint
    initialize_pi0_remote(cfg.pi0_api_url, cfg.pretrained_checkpoint)

    # Initialize SIMPLER task suite
    task_suite = get_benchmark(cfg.task_suite_name)()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Load pre-generated rephrases if needed
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        seeds = itertools.count(1000)

        # Initialize LIBERO environment and task description
        env = task_suite.get_env(task) if hasattr(task_suite, 'get_env') else None
        if env is None:
            from experiments.robot.simpler.simpler_utils_robomonkey import get_simpler_env as _get_env
            env = _get_env(task, cfg.model_family)
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

            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            elif cfg.task_suite_name.startswith("simpler"):
                max_steps = 150
            else:
                raise NotImplementedError

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            # Create progress bar for each episode
            pbar = tqdm.tqdm(total=max_steps + cfg.num_steps_wait, desc=f"Episode steps")
            while t < max_steps + cfg.num_steps_wait:
                # IMPORTANT: Do nothing for first few timesteps if configured
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
                    t += 1
                    pbar.update(1)
                    continue

                # Get raw image from environment
                raw_img = get_image_from_maniskill2_obs_dict(env, obs)
                replay_images.append(raw_img)

                # For API call, we pass the current single instruction (no local rephrasing here)
                try:
                    api_result = request_actions_from_server(
                        cfg.pi0_api_url,
                        instruction=task_description,
                        image_np=raw_img,
                        state_vec=obs,
                        policy_batch_inference_size=cfg.policy_batch_inference_size,
                        lang_rephrase_num=cfg.lang_rephrase_num,
                    )
                except Exception as e:
                    print(f"Error calling PI0 API: {e}")
                    # Fallback to no-op step
                    obs, reward, done, trunc, info = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
                    t += 1
                    pbar.update(1)
                    continue

                # Parse action; take the first step of the predicted chunk for this timestep
                actions = api_result.get('actions', [])
                if len(actions) == 0:
                    # Fallback if server returned nothing
                    next_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                else:
                    first = actions[0]
                    # Normalize to shape (7,)
                    next_action = np.array(first, dtype=np.float32).reshape(-1)
                    if next_action.shape[0] != 7:
                        # If server returned [[...]]
                        next_action = np.array(first, dtype=np.float32).reshape(-1)
                        if next_action.shape[0] != 7:
                            raise ValueError(f"Invalid action shape from server: {np.array(first).shape}")

                # Log placeholders for parity
                episode_data['verifier_scores'].append(None)
                episode_data['selected_instructions'].append(task_description)
                episode_data['execute_actions'].append(next_action.copy())
                episode_data['step_timestamps'].append(t)

                # Step environment
                obs, reward, done, trunc, info = env.step(next_action)
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
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
                model_name=str(cfg.pretrained_checkpoint),
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=cfg.policy_batch_inference_size,
                log_file=log_file
            )

            # Save episode data to pickle file
            save_episode_data_openpi(
                episode_data, total_episodes, success=done,
                task_description=original_task_description,
                transformation_type=cfg.lang_transform_type,
                model_name=str(cfg.pretrained_checkpoint),
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=cfg.policy_batch_inference_size,
                log_file=log_file
            )

            # Save at most 5 successes and at most 5 failures
            if cfg.use_wandb and ((done and task_successes < 5) or (not done and task_episodes - task_successes < 5)):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log({f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))})

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


