"""
run_simpler_eval_with_verifier.py

Runs a model in a SimplerEnv environment with batch language instruction processing
and VLA-CLIP trajectory-based action verification.

Usage:
    # OpenVLA with batch verifier:
    python experiments/robot/simpler/run_simpler_eval_with_verifier.py \
        --model_family openvla \
        --pretrained_checkpoint <VLA_CHECKPOINT_PATH> \
        --task_suite_name [ simpler_widowx ... ] \
        # --- Batch Language Verifier Args ---
        --use_batch_verifier True \
        --batch_server_url http://localhost:3200 \
        # --- Trajectory VLA-CLIP Scorer Args ---
        --use_vla_clip_trajectory_scorer True \
        --vla_clip_traj_model_path <TRAJ_CLIP_CHECKPOINT_PATH> \
        --vla_clip_history_length <HISTORY_LENGTH> \
        --vla_clip_use_transformer [ True | False ] \
        # --- Language Generation Args ---
        --clip_select_action_num_candidates 3 \
        --clip_select_action_strategy [ highest_score | softmax_sample ] \
        --vla_clip_score_threshold 0.5 \
        --lang_transform_type [ rephrase | no_transform ] \
        # --- Other Args ---
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import requests
import json_numpy as json

import draccus
import numpy as np
import tqdm
import collections
import imageio
import wandb
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.simpler_utils import (
    convert_maniskill,
    get_simpler_dummy_action,
    get_simpler_env,
    get_simpler_img,
)

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import (
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)

# Import VLA-CLIP inference for trajectory scoring
sys.path.append("/root/vla-clip/bridge_verifier")
from vla_clip_inference_bridge import VLA_CLIP_Bridge_Inference, ACTION_PADDING_VALUE
sys.path.append("/root/vla-clip/clip_verifier/scripts")
from lang_transform import LangTransform

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)

    #################################################################################################################
    # SimplerEnv environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "simpler_widowx"          # Task suite.
    initial_states_type: str = "eval"
    num_steps_wait: int = 0                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Batch Language Verifier Parameters
    #################################################################################################################
    use_batch_verifier: bool = True                 # Enable the batch language verifier?
    batch_server_url: str = "http://localhost:3200"  # URL of the SGLang batch server
    batch_temperature: float = 0.2                   # Temperature for batch inference

    #################################################################################################################
    # Trajectory VLA-CLIP/DINO Scorer (Optional)
    #################################################################################################################
    use_vla_clip_trajectory_scorer: bool = False     # Enable the trajectory scorer?
    use_vla_dino_trajectory_scorer: bool = False     # Enable the DINO trajectory scorer?
    vla_clip_traj_model_path: Optional[str] = None   # Path to the trajectory VLA-CLIP/DINO model
    vla_clip_history_length: int = 10                # History length (MUST match model training)
    vla_clip_use_transformer: bool = True            # Does the trajectory model use a transformer?
    clip_select_action_num_candidates: int = 3       # Number of candidate instructions for action selection
    clip_select_action_strategy: str = "highest_score"  # Strategy: 'highest_score' or 'softmax_sample'
    vla_clip_score_threshold: float = 0.5            # Threshold to trigger candidate generation/evaluation

    #################################################################################################################
    # Language Transformation Parameters
    #################################################################################################################
    lang_transform_type: str = "rephrase"            # Type of language transformation
    use_original_task_description: bool = False      # Use original task description for scoring

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    generate_rephrases: bool = False

def get_batch_actions(instructions: List[str], image_path: str, server_url: str, temperature: float = 1.0):
    """
    Get batch actions for multiple instructions using the SGLang batch server.
    
    Args:
        instructions: List of instruction strings
        image_path: Path to the image file
        server_url: URL of the batch server
        temperature: Temperature for sampling
    
    Returns:
        Tuple of (output_ids, actions) as numpy arrays
    """
    image_path = os.path.abspath(image_path)
    
    payload = {
        "instructions": instructions,
        "image_path": image_path,
        "temperature": temperature
    }

    try:
        res = requests.post(
            f"{server_url}/batch",
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        res.raise_for_status()
        result = json.loads(res.text)
        return np.array(result["output_ids"]), np.array(result["actions"])
    except Exception as e:
        print(f"Error calling batch server: {e}")
        return None, None

def load_rephrases(task_suite_name: str):
    """Load pre-generated rephrases for the task suite."""
    # Make the path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'simpler_rephrased_topped_up.json')
    
    try:
        with open(json_path, 'r') as f:
            all_rephrases = json.load(f)
        # The new format has an "instructions" key containing the task data
        return all_rephrases.get("instructions", {})
    except FileNotFoundError:
        print(f"Warning: Could not find rephrase file {json_path}. Using empty rephrases.")
        return {}


@draccus.wrap()
def eval_simpler_with_verifier(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    if cfg.use_vla_clip_trajectory_scorer or cfg.use_vla_dino_trajectory_scorer:
        assert cfg.vla_clip_traj_model_path is not None, "If using scorer, `vla_clip_traj_model_path` must be specified."
        assert cfg.vla_clip_history_length > 0, "`vla_clip_history_length` must be positive."
        assert cfg.vla_clip_score_threshold is not None, "`vla_clip_score_threshold` must be specified when using scorer."
        print(f"Using Trajectory VLA-CLIP Scorer: H={cfg.vla_clip_history_length}, Transformer={cfg.vla_clip_use_transformer}, Threshold={cfg.vla_clip_score_threshold}")
    else:
        print("Trajectory VLA Scorer DISABLED.")

    if cfg.use_batch_verifier:
        print(f"Using Batch Language Verifier: {cfg.batch_server_url}")
    else:
        print("Batch Language Verifier DISABLED.")

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    else:
        cfg.unnorm_key = "bridge_orig"

    # Load model
    # model = get_model(cfg)

    # # [OpenVLA] Get Hugging Face processor
    # processor = None
    # if cfg.model_family == "openvla":
    #     processor = get_processor(cfg)

    # Initialize VLA-CLIP scorer
    vla_clip_scorer = None
    action_dim = 7
    if cfg.use_vla_clip_trajectory_scorer:
        vla_clip_scorer = VLA_CLIP_Bridge_Inference(
            model_path=cfg.vla_clip_traj_model_path,
            history_length=cfg.vla_clip_history_length,
            use_transformer=cfg.vla_clip_use_transformer
        )
        if hasattr(vla_clip_scorer.model, 'action_dim'):
             action_dim = vla_clip_scorer.model.action_dim
             print(f"Inferred action_dim={action_dim} from VLA-CLIP Scorer.")
        else:
             print(f"Could not infer action_dim from VLA-CLIP Scorer, using default {action_dim}.")

    # Initialize language transformer
    lang_transform = LangTransform()

    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.use_batch_verifier:
        run_id += "-BatchVerifier"
    if cfg.use_vla_clip_trajectory_scorer:
        run_id += f"-VLACLIPTraj_{Path(cfg.vla_clip_traj_model_path).stem}"
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
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Load pre-generated rephrases if available
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)

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

        # Initialize environment and task description depending on suite
        if cfg.task_suite_name.startswith("simpler"):
            env = get_simpler_env(task)
        else:
            # Uses benchmark's factory (e.g., RL4VLA suites via locally registered envs)
            env = task_suite.make(task)
        original_task_description = env.get_language_instruction()

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc="Trials", leave=False):
            print(f"\nTask: {original_task_description} (Trial {episode_idx + 1}/{cfg.num_trials_per_task})")
            log_file.write(f"\nTask: {original_task_description} (Trial {episode_idx + 1}/{cfg.num_trials_per_task})\n")

            # Reset environment to specified seed (initial state)
            obs, reset_info = env.reset(seed=next(seeds))

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            executed_action_history = collections.deque(maxlen=cfg.vla_clip_history_length)
            padding_action_vector = np.full(action_dim, ACTION_PADDING_VALUE, dtype=np.float32)

            if cfg.task_suite_name.startswith("simpler"):
                max_steps = 150  # data is at 5Hz, so this is 30 seconds
            else:
                max_steps = 400

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            pbar = tqdm.tqdm(total=max_steps, desc="Episode Progress")

            all_scores = []
            all_actions = []
            all_selected_instructions = []

            # Generate language instructions for this task
            if cfg.lang_transform_type == "no_transform":
                candidate_instructions = [original_task_description]
            else:
                # Use pre-generated rephrases if available, otherwise generate on-the-fly
                # Find the task_id that matches the original_task_description
                matching_task_id = None
                for task_key, task_data in preloaded_rephrases.items():
                    if task_key == original_task_description:
                        matching_task_id = task_key
                        break
                if matching_task_id is not None:
                    rephrased_list = preloaded_rephrases[matching_task_id]["rephrases"]
                    candidate_instructions = rephrased_list[:cfg.clip_select_action_num_candidates]
                else:
                    print(f"No preloaded rephrases found for task: {original_task_description}")
                    raise ValueError(f"No preloaded rephrases found for task: {original_task_description}")
                if cfg.generate_rephrases and matching_task_id is not None:
                    # Generate rephrases on-the-fly
                    rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases_easy"]
                    fake_input_language_instruction = rephrased_list[0]
                    candidate_instructions = [fake_input_language_instruction]
                    if cfg.clip_select_action_num_candidates > 1:
                        additional_instructions = lang_transform.transform(
                            fake_input_language_instruction, 
                            cfg.lang_transform_type, 
                            batch_number=cfg.clip_select_action_num_candidates-1
                        )
                        candidate_instructions.extend(additional_instructions)
            log_file.write(f"Candidate instructions: {candidate_instructions}\n")
            
            while t < max_steps + cfg.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(get_simpler_dummy_action(cfg.model_family))
                    t += 1
                    pbar.update(1)
                    continue

                # Get preprocessed image
                get_simpler_img(env, obs, resize_size)
                img_verifier = get_simpler_img(env, obs, resize_size, verifier=True)
                # Save preprocessed image for replay video
                image_path = f"./transfer_images/reward_img.jpg"
                replay_images.append(imageio.imread(image_path))

                # use_wrist_image
                if cfg.use_wrist_image:
                    raise NotImplementedError

                # buffering #obs_history images, optionally
                image_history = replay_images[-cfg.obs_history :]
                if len(image_history) < cfg.obs_history:
                    image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                # same but for optional wrist images
                if cfg.use_wrist_image:
                    wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                    if len(wrist_image_history) < cfg.obs_history:
                        wrist_image_history.extend(
                            [replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history))
                        )
                    # interleaved images [... image_t, wrist_t ...]
                    image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

                # Prepare observations dict
                observation = {
                    "full_image": image_history,
                    "state": obs["extra"]["tcp_pose"],
                }

                # --- Action Generation and VLA-CLIP Scoring ---
                action_to_execute = None
                current_vla_clip_score = np.nan
                current_history_for_scoring = None

                # Use batch server for multiple instructions
                temp_image_path = f"./transfer_images/reward_img.jpg"
                output_ids, actions = get_batch_actions(
                    candidate_instructions, 
                    temp_image_path, 
                    cfg.batch_server_url, 
                    cfg.batch_temperature
                )
                
                if actions is not None and len(actions) == len(candidate_instructions):
                    predicted_actions = [convert_maniskill(action) for action in actions]

                # Default action is the first one
                action_to_execute = predicted_actions[0]

                # Score actions if VLA-CLIP scorer is enabled
                if vla_clip_scorer:
                    # Prepare all padded histories
                    padded_histories = []
                    for action in predicted_actions:
                        hist_list = list(executed_action_history)
                        H = cfg.vla_clip_history_length
                        num_pad = H - len(hist_list) - 1
                        past = [padding_action_vector] * max(0, num_pad) + hist_list[-(H - 1):]
                        padded_history = np.array(past + [action.copy()], dtype=np.float32)
                        padded_histories.append(padded_history)
                    
                    # Get instructions for each action
                    if cfg.use_original_task_description:
                        instructions = [original_task_description] * len(predicted_actions)
                    else:
                        instructions = candidate_instructions[0]
                    
                    # Use batch scoring for efficiency
                    scores = vla_clip_scorer.get_history_score(
                        img_verifier,
                        instructions,
                        padded_histories
                    )
                    # Select the best action based on strategy
                    scores = np.array(scores)
                    if cfg.clip_select_action_num_candidates > 1:
                        if cfg.clip_select_action_strategy == "highest_score":
                            valid_indices = np.where(scores > -np.inf)[0]
                            if len(valid_indices) == 0:
                                print("  Warning: All candidate scores are invalid (-inf). Using first action.")
                                current_vla_clip_score = -np.inf
                                current_history_for_scoring = padded_histories[0]
                            else:
                                scores_valid = scores[valid_indices]
                                best_valid_idx_in_valid_list = np.argmax(scores_valid)
                                best_candidate_idx = valid_indices[best_valid_idx_in_valid_list]
                                all_selected_instructions.append(candidate_instructions[best_candidate_idx])
                                action_to_execute = predicted_actions[best_candidate_idx]
                                current_vla_clip_score = scores[best_candidate_idx]
                                current_history_for_scoring = padded_histories[best_candidate_idx]
                                if best_candidate_idx != 0:
                                    print(f"  [t={t}] Selected alternative action via: '{candidate_instructions[best_candidate_idx]}' (Score: {current_vla_clip_score:.3f})")
                                else:
                                    print(f"  [t={t}] Kept original action (Score: {current_vla_clip_score:.3f}) after evaluating alternatives.")
                        elif cfg.clip_select_action_strategy == "softmax_sample":
                            # Implement softmax sampling if needed
                            valid_indices = np.where(scores > -np.inf)[0]
                            if len(valid_indices) == 0:
                                print("  Warning: All candidate scores are invalid (-inf). Using first action.")
                                action_to_execute = predicted_actions[0]
                                current_vla_clip_score = -np.inf
                                current_history_for_scoring = padded_histories[0]
                            else:
                                scores_valid = scores[valid_indices]
                                probs = np.exp(scores_valid) / np.sum(np.exp(scores_valid))
                                selected_idx = np.random.choice(len(valid_indices), p=probs)
                                best_candidate_idx = valid_indices[selected_idx]
                                all_selected_instructions.append(candidate_instructions[best_candidate_idx])
                                action_to_execute = predicted_actions[best_candidate_idx]
                                current_vla_clip_score = scores[best_candidate_idx]
                                current_history_for_scoring = padded_histories[best_candidate_idx]
                    else:
                        current_vla_clip_score = scores[0]
                        current_history_for_scoring = padded_histories[0]

                # --- Execute Action and Update State ---
                obs, reward, done, trunc, info = env.step(action_to_execute)
                
                # IMPORTANT: Append the action *actually executed* to the history
                executed_action_history.append(np.array(action_to_execute))

                # Append the score of the executed action for logging
                all_scores.append(current_vla_clip_score if not np.isnan(current_vla_clip_score) and current_vla_clip_score > -np.inf else np.nan)
                all_actions.append(action_to_execute)
                
                if done:
                    task_successes += 1
                    total_successes += 1
                    pbar.update(max_steps-t)
                    break
                t += 1
                pbar.update(1)
                if not np.isnan(current_vla_clip_score) and current_vla_clip_score > -np.inf:
                    pbar.set_postfix({"score": f"{current_vla_clip_score:.3f}"})
                else:
                    pbar.set_postfix({"score": "N/A"})

            pbar.close()
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, 
                total_episodes, 
                success=done, 
                transform_type=cfg.lang_transform_type,
                task_description=original_task_description, 
                log_file=log_file,
                score_list=all_scores,
                action_list=all_actions,
                task_description_list=all_selected_instructions,
                clip_update_num=cfg.clip_select_action_num_candidates
            )

            # Save at most 5 successes and at most 5 failures
            if cfg.use_wandb and ((done and task_successes < 5) or (not done and task_episodes - task_successes < 5)):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log(
                    {f"{original_task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))}
                )

            # Log current results
            avg_score = np.nanmean(all_scores) if all_scores else np.nan
            print(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgStepScore={avg_score:.3f}")
            log_file.write(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgStepScore={avg_score:.3f}\n")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        print(f"Task {task_id} ('{original_task_description}') Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})")
        log_file.write(f"Task {task_id} ('{original_task_description}') Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{original_task_description}": task_success_rate,
                    f"num_episodes/{original_task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print("-" * 30)
    print(f"Overall Success Rate: {total_success_rate:.3f} ({total_successes}/{total_episodes})")
    print("-" * 30)
    
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": total_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_simpler_with_verifier() 