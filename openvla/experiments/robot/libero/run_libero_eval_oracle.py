import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import json
import imageio
import draccus
import numpy as np
from tqdm import tqdm
from libero.libero import benchmark
import collections
import torch
import copy

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
sys.path.append("/root/vla-clip/clip_verifier/scripts")
from lang_transform import LangTransform

torch.set_num_threads(8)

# --- Dataclass and Function Definitions ---
@dataclass
class GenerateConfig:
    # fmt: off

    # --- VLA Model ---
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # --- LIBERO Env ---
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 20

    # --- Oracle Scorer / Action Selection Params ---
    use_oracle_scorer: bool = True                        # Enable the oracle scorer logic
    vla_clip_history_length: int = 10                      # History length for VLA policy / action tracking
    clip_select_action_num_candidates: int = 3             # Number of candidate instructions (incl. current) for action selection
    clip_select_action_strategy: str = "highest_score"     # Strategy: 'highest_score' or 'softmax_sample'
    vla_clip_score_threshold: float = 0.005                 # Threshold for negative L2 norm to trigger candidate evaluation (e.g., if -L2 < -0.1 => L2 > 0.1)

    # --- Logging & Utils ---
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs/libero_evals"
    seed: int = 7
    # Fields for internal use / derived config
    unnorm_key: Optional[str] = None

    # langauge transform
    lang_transform_type: str = "rephrase" 
    use_original_task_description: bool = False

def load_rephrases(task_suite_name):
    # Make the path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'libero_rephrase_pos_rephrase_neg_negation.json')
    with open(json_path, 'r') as f:
        all_rephrases = json.load(f)
    return all_rephrases[task_suite_name]

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "VLA `pretrained_checkpoint` must be specified!"
    if cfg.use_oracle_scorer:
        print(f"Using Oracle Scorer: Candidates={cfg.clip_select_action_num_candidates}, Strategy='{cfg.clip_select_action_strategy}', Threshold={cfg.vla_clip_score_threshold}")
    else:
        print("Oracle Scorer DISABLED. Will use task_description directly.")

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)

    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    run_id = f"EVAL-{cfg.task_suite_name}-VLA_{Path(cfg.pretrained_checkpoint).stem}"
    if cfg.use_oracle_scorer:
        run_id += f"-OracleScorer_cand{cfg.clip_select_action_num_candidates}_strat{cfg.clip_select_action_strategy}_thresh{cfg.vla_clip_score_threshold}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    run_id += f"--{DATE_TIME}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    resize_size = get_image_resize_size(cfg)

    total_episodes, total_successes = 0, 0
    
    lang_transform = LangTransform()
    
    if cfg.task_suite_name == "libero_spatial": max_steps = 220
    elif cfg.task_suite_name == "libero_object": max_steps = 280
    elif cfg.task_suite_name == "libero_goal": max_steps = 300
    elif cfg.task_suite_name == "libero_10": max_steps = 520
    elif cfg.task_suite_name == "libero_90": max_steps = 400
    else: max_steps = 400

    # Load pre-generated rephrases if available
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)

    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(range(cfg.num_trials_per_task), desc="Trials", leave=False):
            env, original_task_description = get_libero_env(task, cfg.model_family, resolution=256)
            # --- Always use the first rephrase as the main instruction, rest as alternatives ---
            rephrased_list = preloaded_rephrases[str(task_id)]["rephrases"]
            if cfg.lang_transform_type == "no_transform":
                task_description = original_task_description
            else:
                task_description = rephrased_list[-1]  # Use the last as the main instruction
            # Comment out on-the-fly generation
            # task_description = lang_transform.transform(original_task_description, cfg.lang_transform_type)
            print(f"\nTask: {task_description} (Trial {episode_idx + 1}/{cfg.num_trials_per_task})")
            log_file.write(f"\nTask: {task_description} (Trial {episode_idx + 1}/{cfg.num_trials_per_task})\n")

            obs = env.reset()
            env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            executed_action_history = collections.deque(maxlen=cfg.vla_clip_history_length)

            log_file.write(f"Starting episode {total_episodes+1}...\n")
            pbar = tqdm(total=max_steps, desc="Episode Progress")

            all_scores = []
            all_actions = []
            all_selected_instructions = []
            # pre_sampled_rephrased_instructions_pool = []
            if cfg.use_oracle_scorer and cfg.clip_select_action_num_candidates > 1:
                pre_sampled_rephrased_instructions_pool = rephrased_list[:-1]  # Use the rest as alternatives
                # Commented out: on-the-fly generation
                # pre_sampled_rephrased_instructions_pool = lang_transform.transform(original_task_description if cfg.use_original_task_description else task_description, 
                #                                                                     cfg.lang_transform_type, 
                #                                                                     batch_number=25)
            while t < max_steps:
                if t < cfg.num_steps_wait:
                    action_to_execute = get_libero_dummy_action(cfg.model_family)
                    obs, reward, done, info = env.step(action_to_execute)
                    t += 1
                    pbar.update(1)
                    continue

                img_for_vla = get_libero_image(obs, resize_size)

                replay_images.append(img_for_vla)
                observation = {"full_image": img_for_vla}

                # --- Action Generation and Oracle Scoring ---
                action_to_execute = None
                current_oracle_score = np.nan # Score is -L2 norm

                # 1. Get reference action from VLA using the *original* task description
                action_true_original_desc = get_action(cfg, model, observation, original_task_description, processor=processor)
                action_true_original_desc = normalize_gripper_action(action_true_original_desc, binarize=True)
                if cfg.model_family == "openvla":
                    action_true_original_desc = invert_gripper_action(action_true_original_desc)

                # 2. Get action using the current `task_description` (might be original or rephrased from previous step)
                action_current_task_desc = get_action(cfg, model, observation, task_description, processor=processor)
                action_current_task_desc = normalize_gripper_action(action_current_task_desc, binarize=True)
                if cfg.model_family == "openvla":
                    action_current_task_desc = invert_gripper_action(action_current_task_desc)
                
                def calculate_oracle_score(current_action, true_original_action):
                    return np.linalg.norm(current_action[:-1] - true_original_action[:-1])
                
                current_oracle_score = calculate_oracle_score(action_current_task_desc, action_true_original_desc)
                action_to_execute = action_current_task_desc # Default action

                # 3. Check threshold and evaluate alternatives if using oracle scorer
                if (cfg.use_oracle_scorer and
                    cfg.clip_select_action_num_candidates > 1 and
                    not np.isnan(current_oracle_score) and
                    current_oracle_score > cfg.vla_clip_score_threshold):

                    print(f"  [t={t}] Current action's oracle score {current_oracle_score:.3f} (L2 diff) > {cfg.vla_clip_score_threshold:.3f}. Evaluating alternatives...")

                    # Initialize lists: first candidate is the current task_description and its action/score
                    candidate_instructions = [task_description]
                    candidate_actions = [action_current_task_desc]
                    oracle_scores = [current_oracle_score]

                    # Generate additional candidate instructions by rephrasing current task_description
                    num_additional_to_generate = cfg.clip_select_action_num_candidates - 1
                    if num_additional_to_generate > 0 and pre_sampled_rephrased_instructions_pool:
                        # Using pre_sampled_rephrased_instructions_pool based on original_task_description:
                        # sample_indices = np.random.choice(len(pre_sampled_rephrased_instructions_pool), size=num_additional_to_generate, replace=False)
                        sample_indices = np.arange(len(pre_sampled_rephrased_instructions_pool))[:num_additional_to_generate]
                        additional_rephrased_instr = [pre_sampled_rephrased_instructions_pool[i] for i in sample_indices]
                        candidate_instructions.extend(additional_rephrased_instr)

                    # Evaluate additional candidates (if any were added)
                    # Start from index 1 because index 0 (current_task_desc) is already processed
                    for i in range(1, len(candidate_instructions)):
                        instr_c = candidate_instructions[i]
                        action_c = get_action(cfg, model, copy.deepcopy(observation), instr_c, processor=processor)
                        action_c = normalize_gripper_action(action_c, binarize=True)
                        if cfg.model_family == "openvla":
                            action_c = invert_gripper_action(action_c)
                        
                        score_c = calculate_oracle_score(action_c, action_true_original_desc)
                        
                        assert i >= len(candidate_actions), "Should happen if we extended candidate_instructions"
                        candidate_actions.append(action_c)
                        oracle_scores.append(score_c)


                    oracle_scores_np = np.array(oracle_scores).squeeze()
                    
                    # print candidate_instructions and the corresponding oracle_scores
                    # for i in range(len(candidate_instructions)):
                    #     print(f"  Candidate {i}: {candidate_instructions[i]} (Score: {oracle_scores_np[i]:.3f})")
                    # input()

                    if cfg.clip_select_action_strategy == "highest_score":
                        valid_indices = np.where(oracle_scores_np > -np.inf)[0] # Check for valid scores
                        if len(valid_indices) == 0:
                            print("  Warning: All candidate oracle scores are invalid. Using current task_description action.")
                            chosen_candidate_idx = 0 # Default to current task_description
                        else:
                            scores_to_consider = oracle_scores_np[valid_indices]
                            best_idx_in_valid = np.argmin(scores_to_consider)
                            chosen_candidate_idx = valid_indices[best_idx_in_valid]
                            all_selected_instructions.append(candidate_instructions[chosen_candidate_idx])
                    
                    action_to_execute = candidate_actions[chosen_candidate_idx]
                    current_oracle_score = oracle_scores_np[chosen_candidate_idx]
                    
                    # Update task_description if a different instruction was chosen
                    if chosen_candidate_idx != 0: # 0 is the original `task_description` for this step
                        task_description = candidate_instructions[chosen_candidate_idx]
                        print(f"  [t={t}] Oracle selected alternative action via: '{task_description}' (Action Diff: {current_oracle_score:.3f})")
                    else:
                        print(f"  [t={t}] Oracle kept current action via: '{task_description}' (Action Diff: {current_oracle_score:.3f}) after evaluating alternatives.")

                # --- Execute Action and Update State ---
                action_to_execute_list = action_to_execute.tolist()
                obs, reward, done, info = env.step(action_to_execute_list)
                # IMPORTANT: Append the action *actually executed* to the history
                executed_action_history.append(np.array(action_to_execute_list))

                # Append the score of the executed action for logging
                all_scores.append(current_oracle_score if not np.isnan(current_oracle_score) and current_oracle_score > -np.inf else np.nan)
                all_actions.append(action_to_execute)

                if done:
                    task_successes += 1
                    total_successes += 1
                    pbar.update(max_steps-t)
                    break
                t += 1
                pbar.update(1)
                if not np.isnan(current_oracle_score) and current_oracle_score > -np.inf:
                    pbar.set_postfix({"L2_diff": f"{current_oracle_score:.3f}"}) # Display L2 difference
                else:
                    pbar.set_postfix({"L2_diff": "N/A"})

            pbar.close()
            task_episodes += 1
            total_episodes += 1

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
                clip_update_num=cfg.clip_select_action_num_candidates,
                use_original_task_description=cfg.use_original_task_description,
                oracle_scorer=cfg.use_oracle_scorer,
            )

            avg_score = np.nanmean(all_scores) if all_scores else np.nan
            print(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgStepScore={avg_score:.3f}")
            log_file.write(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgStepScore={avg_score:.3f}\n")
            log_file.flush()

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        print(f"Task {task_id} ('{original_task_description}') Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})")
        log_file.write(f"Task {task_id} ('{original_task_description}') Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})\n")
       
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print("-" * 30)
    print(f"Overall Success Rate: {total_success_rate:.3f} ({total_successes}/{total_episodes})")
    log_file.write(f"\nOverall Success Rate: {total_success_rate:.3f} ({total_successes}/{total_episodes})\n")
    print("-" * 30)

    log_file.close()



if __name__ == "__main__":
    eval_libero()
