"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
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
    resize_image,
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
from vla_clip_inference import VLA_CLIP_Inference
from lang_transform import LangTransform

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 3                  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    
    clip_model_path: str = "/home/xilun/vla-comp/clip_verifier/bash/model_checkpoints/spatial_clip_action_encoder_only_augmented_dataset_epoch_100.pt"

    language_transformation_type: str = "no_transform"
    
    alignment_text: str = "transformed" # "original" or "transformed"

    # Gradient optimization
    use_gradient_optimization: bool = False          # Whether to use gradient-based optimization
    optimize_both_action_and_text: bool = False      # Whether to optimize both action and text
    optimization_iterations: int = 50               # Number of optimization iterations
    optimization_lr: float = 1e-3                    # Learning rate
    optimization_reg_weight: float = 0.1            # Regularization weight
    topk: int = 1
    
    # Sampling-based optimization
    sampling_based_optimization: bool = False      # Whether to use sampling-based optimization
    clip_action_iter: int = 1
    beta: float = 0.05


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    
    # check assertion on optimization 
    assert not (cfg.use_gradient_optimization and cfg.sampling_based_optimization), "Cannot use both gradient optimization and sampling-based optimization!"

    # Load model
    model = get_model(cfg)
    
    clip_inference_model = VLA_CLIP_Inference(cfg.clip_model_path, trajectory_mode = False)
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    if cfg.use_gradient_optimization:
        run_id = f"EVAL-{cfg.task_suite_name}-{cfg.language_transformation_type}-topk_{cfg.topk}-beta_{cfg.beta}-alignment_{cfg.alignment_text}-gradient_optimization_{cfg.use_gradient_optimization}-update_text_{cfg.optimize_both_action_and_text}"
    else:
        run_id = f"EVAL-{cfg.task_suite_name}-{cfg.language_transformation_type}-action_iter_{cfg.clip_action_iter}-beta_{cfg.beta}-alignment_{cfg.alignment_text}"
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

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    language_transform = LangTransform()

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(range(cfg.num_trials_per_task)):
            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256, task_seed=0)
            original_task_description = task_description
            print(f"\nOriginal Task: {task_description}")
            log_file.write(f"\nOriginal Task: {task_description}\n")
            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            reword_img = get_libero_image(obs, resize_size)
            task_description = language_transform.transform(task_description, cfg.language_transformation_type, 1)
            first_task_description = task_description
            print(f"\nTransformed Task: {first_task_description}")
            log_file.write(f"\nTransformed Task: {first_task_description}\n")
            if cfg.sampling_based_optimization:
                task_description = language_transform.transform(first_task_description, cfg.language_transformation_type, cfg.clip_action_iter, reword_img)
                length = len(task_description)
                counter = 0
                while length != cfg.clip_action_iter:
                    print ("Generated task description is not valid, generating again...")
                    task_description = language_transform.transform(first_task_description, cfg.language_transformation_type, cfg.clip_action_iter, reword_img)
                    length = len(task_description)
                    counter += 1
                    if counter > 5:
                        print ("Failed to generate valid task description after 5 attempts, exiting...")
                        break
                    
                print(f"\nSynthesized Task: {task_description}")
                log_file.write(f"\nSynthesized Task: {task_description}\n")
                
            # Setup
            t = 0
            replay_images = []
            score_list = []
            action_list = []
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

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            with tqdm(total=max_steps + cfg.num_steps_wait, desc="Episode Progress") as pbar:
                while t < max_steps + cfg.num_steps_wait:
                    # try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        pbar.update(1)
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                        
                    clip_img = resize_image(obs["agentview_image"], (128, 128))
                    
                    aligned_task_description = original_task_description if cfg.alignment_text == "original" else first_task_description
                     
                    # Three distinct optimization paths
                    if cfg.use_gradient_optimization:
                        if cfg.optimize_both_action_and_text:
                            vla_action = get_action(
                                cfg,
                                model,
                                observation,
                                first_task_description,
                                processor=processor,
                            )
                            vla_action = normalize_gripper_action(vla_action, binarize=True)
                            if cfg.model_family == "openvla":
                                vla_action = invert_gripper_action(vla_action)
                                
                            optimized_action, optimized_text, best_score = clip_inference_model.optimize_action_and_instruction(
                                clip_img, 
                                aligned_task_description, 
                                num_iterations=cfg.optimization_iterations, 
                                lr=cfg.optimization_lr, 
                                reg_weight=cfg.optimization_reg_weight,
                                topk=cfg.topk,
                                vla_action=vla_action
                            )
                                
                            action = normalize_gripper_action(optimized_action, binarize=True)
                            if not isinstance(action, torch.Tensor):
                                action = torch.tensor(action)

                        else:
                            vla_action = get_action(
                                cfg,
                                model,
                                observation,
                                first_task_description,
                                processor=processor,
                            )
                            
                            vla_action = normalize_gripper_action(vla_action, binarize=True)
                            if cfg.model_family == "openvla":
                                vla_action = invert_gripper_action(vla_action)

                            action, best_score = clip_inference_model.optimize_action_gradient(
                                clip_img, 
                                aligned_task_description, 
                                vla_action=[vla_action],
                                num_iterations=cfg.optimization_iterations, 
                                lr=cfg.optimization_lr, 
                                reg_weight=cfg.optimization_reg_weight,
                                topk=cfg.topk,
                                beta=cfg.beta,
                            )

                        score_list.append(best_score)
                        action_list.append(action)
                        
                    elif cfg.sampling_based_optimization:
                        # Use sampling-based optimization (existing method)
                        iteration_action_list = []
                        task_description_list = []
                        
                        # Generate multiple samples
                        for i in range(cfg.clip_action_iter):
                            iteration_action = get_action(
                                cfg,
                                model,
                                observation,
                                task_description[i],
                                processor=processor,
                            )
                            iteration_action = normalize_gripper_action(iteration_action, binarize=True)
                            if cfg.model_family == "openvla":
                                iteration_action = invert_gripper_action(iteration_action)
                            if not isinstance(iteration_action, torch.Tensor):
                                iteration_action = torch.tensor(iteration_action)
                        
                            iteration_action_list.append(iteration_action)
                            task_description_list.append(task_description[i])
                        
       
                        # Filter using CLIP scores
                        iteration_image_logits, action, predicted_task_description = clip_inference_model.online_predict(
                            clip_img, 
                            aligned_task_description, 
                            iteration_action_list, 
                            task_description_list,
                            softmax=True, 
                            beta=cfg.beta
                        )
                        score_list.append(iteration_image_logits)
                        action_list.append(action)
                        
                    else:
                        # No optimization, just use the base action
                        action = get_action(
                            cfg,
                            model,
                            observation,
                            first_task_description,
                            processor=processor,
                        )
                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)
                        if not isinstance(action, torch.Tensor):
                            action = torch.tensor(action)

                        image_logits, _, _ = clip_inference_model.online_predict(
                            clip_img, 
                            aligned_task_description, 
                            [action]
                        )
                        score_list.append(image_logits)
                        action_list.append(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    pbar.update(1)

                    # except Exception as e:
                    #     print(f"Caught exception: {e}")
                    #     log_file.write(f"Caught exception: {e}\n")
                    #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            # save_rollout_video(
            #     replay_images, 
            #     total_episodes, 
            #     success=done, 
            #     task_description=aligned_task_description, 
            #     log_file=log_file,
            #     score_list=score_list,
            #     action_list=action_list,
            #     language_transformation_type=cfg.language_transformation_type,
            #     sampling_based_optimization=cfg.sampling_based_optimization,
            #     gradient_based_optimization=cfg.use_gradient_optimization,
            # )

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
    eval_libero()
