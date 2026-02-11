"""
SIMPLER Environment Evaluation with OpenPI and RoboMonkey Verifier

This script evaluates vision-language-action policies on SIMPLER benchmark tasks using:
- PI0 policy with batch inference
- Language instruction rephrasing
- Action verification and selection using ensemble model
- Multi-step action chunking

For paper: "RoboMonkey: Improving Robot Manipulation through Language Instruction Verification"
"""

import itertools
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
import wandb

# SIMPLER environment imports
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.eval_utils import (
    convert_maniskill_with_bridge_adapter,
    create_bridge_adapter_wrapper,
    get_simpler_dummy_action,
    get_simpler_env,
    load_rephrases,
    process_inputs,
    process_raw_image_to_jpg,
    save_episode_data_openpi,
    save_rollout_video_openpi,
    set_seed_everywhere,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# PI0 policy imports (installed as module via env_simpler_pi.sh)
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

# Ensemble verifier imports (installed as module via env_simpler_pi.sh)
from bridge_verifier.ensemble_eval import EfficientEnsembleMerged

# Constants
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


# =========================================================================================
# Configuration
# =========================================================================================

@dataclass
class GenerateConfig:
    """Configuration for SIMPLER evaluation with PI0 and RoboMonkey verifier."""
    
    # Model parameters
    model_family: str = "openvla"
    hf_token: str = Path(".hf_token")
    pretrained_checkpoint: Union[str, Path] = "juexzz/INTACT-pi0-finetune-rephrase-bridge"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    obs_history: int = 1
    
    # Environment parameters
    task_suite_name: str = "simpler_widowx"
    initial_states_type: str = "eval"
    num_steps_wait: int = 0
    num_trials_per_task: int = 300
    
    # Logging parameters
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    prefix: str = ''
    use_wandb: bool = False
    wandb_project: str = "prismatic"
    wandb_entity: Optional[str] = None
    seed: int = 7
    
    # RoboMonkey verifier parameters
    use_verifier: bool = True
    initial_samples: int = 5
    augmented_samples: int = 32
    
    # Action chunking parameters
    n_action_steps: int = 4
    action_ensemble_temp: float = -0.8
    
    # Language transformation parameters
    lang_transform_type: str = "no_transform"
    lang_rephrase_num: int = 8
    
    # Batch inference parameters
    policy_batch_inference_size: int = 2

# =========================================================================================
# Main Evaluation Function
# =========================================================================================

@draccus.wrap()
def eval_simpler(cfg: GenerateConfig) -> None:
    """Main evaluation function for SIMPLER benchmark with PI0 and RoboMonkey.
    
    Args:
        cfg: Configuration object containing all evaluation parameters
    """
    # Validate configuration
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.initial_samples > 0, "Invalid initial_samples: should be > 0"
    assert cfg.augmented_samples > 0, "Invalid augmented_samples: should be > 0"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Set action un-normalization key
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    else:
        cfg.unnorm_key = "bridge_orig"

    # Initialize logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize PI0 policy
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    pi0_policy = PI0Policy.from_pretrained(cfg.pretrained_checkpoint)
    
    if torch.cuda.is_available():
        pi0_policy.to("cuda")
        pi0_policy.config.device = "cuda"
    
    pi0_policy.config.n_action_steps = int(cfg.n_action_steps)
    print(f"PI0Policy device: {pi0_policy.config.device}")
    
    # Initialize verifier model
    if cfg.use_verifier:
        print("Loading ensemble model for similarity scoring...")
        # Use dynamic path relative to the VLA-CLIP root
        vla_clip_root = Path(__file__).resolve().parents[5]
        ensemble_checkpoint_path = vla_clip_root / "bridge_verifier" / "cover_verifier_bridge.pt"
        ensemble_model = EfficientEnsembleMerged(str(ensemble_checkpoint_path))
        print("Ensemble model loaded successfully!")
    else:
        ensemble_model = None
    
    # Initialize SIMPLER task suite

    task_suite = get_benchmark(cfg.task_suite_name)()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    
    # Load pre-generated rephrases
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)
    
    # Create adapter for preprocessing
    action_queue = None
    if not hasattr(pi0_policy, '_preprocess_adapter'):
        pi0_policy._preprocess_adapter = create_bridge_adapter_wrapper(cfg.action_ensemble_temp)
    preprocess_adapter = pi0_policy._preprocess_adapter
    
    # Action noise for batch inference
    action_noise_std = 1.0

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        seeds = itertools.count(1000)

        # Initialize environment and task description
        env = get_simpler_env(task, cfg.model_family)
        original_task_description = env.get_language_instruction()
        
        # Load rephrased instructions if using language transformation
        if cfg.lang_transform_type == "no_transform":
            assert cfg.lang_rephrase_num == 1, "Language rephrase number must be 1 for no transformation"
            task_description = original_task_description
            rephrased_list = None
            matching_task_id = None
        else:
            # Find matching task in preloaded rephrases
            matching_task_id = None
            for task_key, task_data in preloaded_rephrases.items():
                if task_key == original_task_description:
                    matching_task_id = task_key
                    break
            
            if matching_task_id is not None:
                rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"][:cfg.lang_rephrase_num] 
            else:
                raise ValueError(f"No preloaded rephrases found for task: {original_task_description}")

        # Run episodes for this task
        task_episodes, task_successes = 0, 0
        for trail_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if matching_task_id is not None:
                task_description = preloaded_rephrases[matching_task_id]["original"]
            else:
                task_description = original_task_description
            
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            
            if trail_idx % 50 == 0:
                seeds = itertools.count(1000)
            
            obs, reset_info = env.reset(seed=next(seeds))

            # Initialize episode
            t = 0
            replay_images = []
            action_history = []
            
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
            
            # Set max steps based on task suite
            if cfg.task_suite_name.startswith("simpler"):
                max_steps = 150
            else:
                raise NotImplementedError

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            pbar = tqdm.tqdm(total=max_steps + cfg.num_steps_wait, desc=f"Episode steps")
            while t < max_steps + cfg.num_steps_wait:
                # Wait for objects to stabilize in simulator
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(get_simpler_dummy_action(cfg.model_family))
                    t += 1
                    pbar.update(1)
                    continue

                # Get raw image from environment
                raw_img = get_image_from_maniskill2_obs_dict(env, obs)
                replay_images.append(raw_img)

                # Buffer observation history
                image_history = replay_images[-cfg.obs_history:]
                if len(image_history) < cfg.obs_history:
                    image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))
                
                # Prepare observations for adapter
                obs_for_adapter = {
                    'observation.images.top': raw_img,
                    'observation.state': obs,
                    'task': task_description
                }
                processed_obs = preprocess_adapter.preprocess(obs_for_adapter)
                
                # Move to policy device
                policy_device = torch.device(pi0_policy.config.device)
                processed_obs = {
                    k: (v.to(device=policy_device) if isinstance(v, torch.Tensor) else v)
                    for k, v in processed_obs.items()
                }
                
                # Get image feature key from policy config
                image_feature_keys = list(pi0_policy.config.image_features.keys())
                image_key = image_feature_keys[0]
                
                # Create batch of language instructions
                batch_size = cfg.policy_batch_inference_size * cfg.lang_rephrase_num
                
                # Build unique instruction list
                if rephrased_list is not None and cfg.lang_rephrase_num > 1:
                    unique_prompts = [task_description] + rephrased_list[:cfg.lang_rephrase_num - 1]
                else:
                    unique_prompts = [task_description]

                # Repeat each instruction for batch inference
                task_list = []
                for p in unique_prompts:
                    task_list.extend([p] * cfg.policy_batch_inference_size)
                    
                assert len(task_list) == batch_size, "Batch size mismatch"
                    
                # Create batch observation dict
                batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
                batch_state = processed_obs['observation.state'].repeat(batch_size, 1)
                
                observation = {
                    image_key: batch_image,
                    "observation.state": batch_state,
                    "task": task_list,
                }
                
                # Call select_action every n_action_steps
                if t % cfg.n_action_steps == 0:
                    with torch.no_grad():
                        output_action_queue = pi0_policy.select_action(observation, noise_std=action_noise_std)
                        action_queue = output_action_queue.copy()
                        output_action_queue.clear()
                
                # Use verifier to select best action
                if cfg.use_verifier and t % cfg.n_action_steps == 0:
                    assert len(action_queue) == cfg.n_action_steps, \
                        f"Action queue length should be {cfg.n_action_steps}, but got {len(action_queue)}"
                    
                    num_past = min(len(action_history), 6)
                    predefined_action_queue = list(action_queue)
                    action_queue.popleft()
                    
                    # Process actions for verifier
                    action_histories_list = process_inputs(
                        batch_size, predefined_action_queue, 
                        verifier_action=True, action_history=action_history.copy(), cfg=cfg
                    )
                    images_list = [process_raw_image_to_jpg(raw_img)] * batch_size
                    
                    with torch.no_grad():
                        # First try with original instruction only (high confidence)
                        max_score, max_instruction, max_action_history, global_action_idx = \
                            ensemble_model.compute_max_similarity_scores_batch(
                                images=images_list[0:1],
                                instructions=[task_description],
                                all_action_histories=action_histories_list[0:1],
                                cfg_repeat_language_instructions=1
                            )
                    
                    # If score is too low, try with all rephrased instructions (low confidence)
                    if max_score < 0.1:
                        with torch.no_grad():
                            max_score, _ , max_action_history, global_action_idx = \
                                ensemble_model.compute_max_similarity_scores_batch(
                                    images=images_list,
                                    instructions=[task_description] * batch_size,
                                    all_action_histories=action_histories_list,
                                    cfg_repeat_language_instructions=cfg.policy_batch_inference_size
                                )
                        # Map global_action_idx back to the corresponding rephrase instruction
                        max_instruction = task_list[global_action_idx]
                    
                    # Get execution-format actions (not verification-format)
                    execution_action_histories_list = process_inputs(
                        batch_size, predefined_action_queue, 
                        verifier_action=False, action_history=action_history.copy(), cfg=cfg
                    )
                    execute_action = execution_action_histories_list[global_action_idx][num_past].copy()
                    
                    # Perform gripper voting
                    group_start = (global_action_idx // cfg.policy_batch_inference_size) * cfg.policy_batch_inference_size
                    group_end = group_start + cfg.policy_batch_inference_size
                    stacked_histories = np.stack(execution_action_histories_list[group_start:group_end])
                    grippers = stacked_histories[:, num_past, -1]
                    
                    # Count votes: >= 0 is closed, < 0 is open
                    close_votes = int((grippers >= 0).sum())
                    open_votes = int((grippers < 0).sum())

                    if close_votes > open_votes:
                        execute_action[-1] = 1.0
                    elif open_votes > close_votes:
                        execute_action[-1] = -1.0
                    else:
                        # Tie: use selected action's sign
                        execute_action[-1] = 1.0 if execute_action[-1] >= 0 else -1.0

                    execute_action[-1] = float(np.sign(execute_action[-1]))
                    
                    # Extract remaining actions from selected batch item
                    selected_action_chunk = deque()
                    for timestep_idx in range(1, cfg.n_action_steps):
                        timestep_actions = predefined_action_queue[timestep_idx]
                        selected_action = timestep_actions[global_action_idx:global_action_idx+1]
                        selected_action_chunk.append(selected_action)
                    
                    action_queue = selected_action_chunk
                    
                    # Store episode data
                    episode_data['verifier_scores'].append(max_score)
                    episode_data['selected_instructions'].append(max_instruction)
                    episode_data['execute_actions'].append(execute_action.copy())
                    episode_data['step_timestamps'].append(t)
                    
                    task_description = max_instruction
                else:
                    # Use actions from queue
                    single_action = action_queue.popleft().cpu().numpy()
                    action_for_env = single_action[0:1]
                    execute_action = convert_maniskill_with_bridge_adapter(
                        action_for_env, verifier_action=False, action_ensemble_temp=cfg.action_ensemble_temp
                    )
                    
                    # Store episode data
                    episode_data['verifier_scores'].append(None)
                    episode_data['selected_instructions'].append(task_description)
                    episode_data['execute_actions'].append(execute_action.copy())
                    episode_data['step_timestamps'].append(t)
                    
                # Update action history for verifier
                if cfg.use_verifier:
                    if t % cfg.n_action_steps == 0:
                        processed_action_for_history = max_action_history[num_past].copy()
                    else:
                        processed_action_for_history = convert_maniskill_with_bridge_adapter(
                            single_action[0:1], verifier_action=True, action_ensemble_temp=cfg.action_ensemble_temp
                        )
                    action_history.append(processed_action_for_history)

                # Execute action in environment
                obs, reward, done, trunc, info = env.step(execute_action)
                
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                
                t += 1
                
                # Update progress bar
                if cfg.use_verifier:
                    pbar.set_description(f"Episode steps (score: {max_score:.3f})")
                pbar.update(1)

            pbar.close()
            task_episodes += 1
            total_episodes += 1

            # Finalize episode data
            episode_data['success'] = done
            episode_data['episode_length'] = t
            action_queue.clear()
            
            # Save rollout video
            save_rollout_video_openpi(
                replay_images, total_episodes, success=done, 
                task_description=original_task_description, 
                transformation_type=cfg.lang_transform_type,
                model_name=cfg.pretrained_checkpoint,
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=cfg.policy_batch_inference_size,
                log_file=log_file
            )
            
            # Save episode data
            save_episode_data_openpi(
                episode_data, total_episodes, success=done, 
                task_description=original_task_description, 
                transformation_type=cfg.lang_transform_type,
                model_name=cfg.pretrained_checkpoint,
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=cfg.policy_batch_inference_size,
                log_file=log_file
            )

            # Log videos to W&B (limit 5 successes and 5 failures per task)
            if cfg.use_wandb and ((done and task_successes < 5) or 
                                 (not done and task_episodes - task_successes < 5)):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                video_array = np.array(replay_images).transpose(0, 3, 1, 2)
                wandb.log({f"{task_description}/{group}/{idx}": wandb.Video(video_array)})

            # Log episode results
            success_rate = total_successes / total_episodes * 100
            print(f"Success: {done}")
            print(f"Episodes: {total_episodes} | Successes: {total_successes} ({success_rate:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Episodes: {total_episodes} | Successes: {total_successes} ({success_rate:.1f}%)\n")
            log_file.flush()

        # Log task results
        task_success_rate = float(task_successes) / float(task_episodes)
        total_success_rate = float(total_successes) / float(total_episodes)
        print(f"\nTask success rate: {task_success_rate:.3f}")
        print(f"Total success rate: {total_success_rate:.3f}")
        log_file.write(f"\nTask success rate: {task_success_rate:.3f}\n")
        log_file.write(f"Total success rate: {total_success_rate:.3f}\n")
        log_file.flush()
        
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            })

    # Finalize logging
    log_file.close()

    # Log final metrics to W&B
    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        })
        wandb.save(local_log_filepath)


# =========================================================================================
# Entry Point
# =========================================================================================

if __name__ == "__main__":
    eval_simpler()
