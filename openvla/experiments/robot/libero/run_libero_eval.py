"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment, optionally scoring actions
using a trajectory-based VLA-CLIP model.

Usage:
    # OpenVLA:
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <VLA_CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        # --- Trajectory VLA-CLIP Args (Optional) ---
        --use_vla_clip_trajectory_scorer True \
        --vla_clip_traj_model_path <TRAJ_CLIP_CHECKPOINT_PATH> \
        --vla_clip_history_length <HISTORY_LENGTH> \
        --vla_clip_use_transformer [ True | False ] \
        # --- Other Args ---
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
from tqdm import tqdm
from libero.libero import benchmark
from PIL import Image
import collections

import wandb
import torch

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
sys.path.append("/home/xilun/vla-clip/clip_verifier/scripts")
from vla_clip_inference import VLA_CLIP_Inference, ACTION_PADDING_VALUE
from lang_transform import LangTransform

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
    num_trials_per_task: int = 5

    # --- Trajectory VLA-CLIP Scorer (Optional) ---
    use_vla_clip_trajectory_scorer: bool = True           # Enable the trajectory scorer?
    vla_clip_traj_model_path: Optional[str] = None         # Path to the trajectory VLA-CLIP model
    vla_clip_history_length: int = 10                      # History length (MUST match model training)
    vla_clip_use_transformer: bool = True                 # Does the trajectory model use a transformer?

    # --- Logging & Utils ---
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs/libero_evals"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7
    # Fields for internal use / derived config
    unnorm_key: Optional[str] = None

    # langauge transform
    lang_transform: str = "rephrase"

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "VLA `pretrained_checkpoint` must be specified!"
    if cfg.use_vla_clip_trajectory_scorer:
        assert cfg.vla_clip_traj_model_path is not None, "If using scorer, `vla_clip_traj_model_path` must be specified."
        assert cfg.vla_clip_history_length > 0, "`vla_clip_history_length` must be positive."
        print(f"Using Trajectory VLA-CLIP Scorer: H={cfg.vla_clip_history_length}, Transformer={cfg.vla_clip_use_transformer}")
    else:
        print("Trajectory VLA-CLIP Scorer DISABLED.")

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

    vla_clip_scorer = None
    action_dim = 7
    if cfg.use_vla_clip_trajectory_scorer:
        vla_clip_scorer = VLA_CLIP_Inference(
            model_path=cfg.vla_clip_traj_model_path,
            history_length=cfg.vla_clip_history_length,
            use_transformer=cfg.vla_clip_use_transformer
        )
        if hasattr(vla_clip_scorer.model, 'action_dim'):
             action_dim = vla_clip_scorer.model.action_dim
             print(f"Inferred action_dim={action_dim} from VLA-CLIP Scorer.")
        else:
             print(f"Could not infer action_dim from VLA-CLIP Scorer, using default {action_dim}.")

    run_id = f"EVAL-{cfg.task_suite_name}-VLA_{Path(cfg.pretrained_checkpoint).stem}"
    if cfg.use_vla_clip_trajectory_scorer:
        run_id += f"-VLACLIPTraj_{Path(cfg.vla_clip_traj_model_path).stem}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    run_id += f"--{DATE_TIME}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
        wandb.config.update(draccus.encode(cfg))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    resize_size = get_image_resize_size(cfg)

    total_episodes, total_successes = 0, 0
    
    # langauge transform
    lang_transform = LangTransform()

    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(range(cfg.num_trials_per_task), desc="Trials", leave=False):
            env, original_task_description = get_libero_env(task, cfg.model_family, resolution=256, task_seed=0)
            task_description = lang_transform.transform(original_task_description, cfg.lang_transform)
            print(f"\nTask: {task_description} (Trial {episode_idx + 1}/{cfg.num_trials_per_task})")
            log_file.write(f"\nTask: {task_description} (Trial {episode_idx + 1}/{cfg.num_trials_per_task})\n")

            obs = env.reset()
            env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            executed_action_history = collections.deque(maxlen=cfg.vla_clip_history_length)
            padding_action_vector = np.full(action_dim, ACTION_PADDING_VALUE, dtype=np.float32)

            if cfg.task_suite_name == "libero_spatial": max_steps = 220
            elif cfg.task_suite_name == "libero_object": max_steps = 280
            elif cfg.task_suite_name == "libero_goal": max_steps = 300
            elif cfg.task_suite_name == "libero_10": max_steps = 520
            elif cfg.task_suite_name == "libero_90": max_steps = 400
            else: max_steps = 400

            log_file.write(f"Starting episode {total_episodes+1}...\n")
            pbar = tqdm(total=max_steps, desc="Episode Progress")

            all_scores = []
            while t < max_steps:
                if t < cfg.num_steps_wait:
                    action_to_execute = get_libero_dummy_action(cfg.model_family)
                    obs, reward, done, info = env.step(action_to_execute)
                    t += 1
                    pbar.update(1)
                    continue

                img_for_vla = get_libero_image(obs, resize_size)
                img_for_clip = get_libero_image(obs, (224, 224))

                replay_images.append(img_for_vla)
                observation = {"full_image": img_for_vla}

                vla_action = get_action(
                    cfg, model, observation, task_description, processor=processor
                )
                vla_action = normalize_gripper_action(vla_action, binarize=True)
                if cfg.model_family == "openvla":
                    vla_action = invert_gripper_action(vla_action)
                action_to_execute = vla_action

                current_vla_clip_score = None
                if cfg.use_vla_clip_trajectory_scorer and vla_clip_scorer:
                    current_hist_len = len(executed_action_history)
                    num_padding = cfg.vla_clip_history_length - current_hist_len

                    padded_history_list = []
                    for _ in range(num_padding):
                        padded_history_list.append(padding_action_vector)
                    padded_history_list.extend(list(executed_action_history))

                    if not padded_history_list:
                        padded_history_list = [padding_action_vector] * cfg.vla_clip_history_length

                    history_for_scoring = np.array(padded_history_list, dtype=np.float32)
                    current_vla_clip_score = vla_clip_scorer.get_history_score(
                        img_for_clip,
                        task_description,
                        history_for_scoring
                    )
                    all_scores.append(current_vla_clip_score)

                action_to_execute_list = action_to_execute.tolist()
                obs, reward, done, info = env.step(action_to_execute_list)
                executed_action_history.append(np.array(action_to_execute_list))

                if done:
                    task_successes += 1
                    total_successes += 1
                    pbar.update(max_steps-t)
                    break
                t += 1
                pbar.update(1)
                if current_vla_clip_score is not None:
                    pbar.set_postfix({"score": f"{current_vla_clip_score:.3f}"})

            pbar.close()
            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                transform_type=cfg.lang_transform,
                task_description=task_description,
                log_file=log_file,
                score_list=all_scores,
                action_list=list(executed_action_history)
            )

            avg_score = np.nanmean(all_scores)
            print(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgScore={avg_score:.3f}")
            log_file.write(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgScore={avg_score:.3f}\n")
            log_file.flush()

        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        print(f"Task {task_id} ('{task_description}') Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})")
        log_file.write(f"Task {task_id} ('{task_description}') Success Rate: {task_success_rate:.2f} ({task_successes}/{task_episodes})\n")
        if cfg.use_wandb:
            wandb.log({f"success_rate_task/{task_description}": task_success_rate}, step=task_id)

    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    print("-" * 30)
    print(f"Overall Success Rate: {total_success_rate:.3f} ({total_successes}/{total_episodes})")
    log_file.write(f"\nOverall Success Rate: {total_success_rate:.3f} ({total_successes}/{total_episodes})\n")
    print("-" * 30)

    log_file.close()

    if cfg.use_wandb:
        wandb.log({"success_rate/total": total_success_rate, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)
        wandb.finish()


if __name__ == "__main__":
    eval_libero()
