import collections
import dataclasses
import logging
import math
import pathlib
import json
import copy
import imageio
import numpy as np
import cv2
import tensorflow as tf
from typing import Optional

# SimplerEnv imports
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# OpenPI imports
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import sys
import os
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SimplerEnv task suites - based on actual RoboMonkey benchmark implementation
SIMPLER_TASK_SUITES = {
    # Main RoboMonkey task suites
    "simpler_widowx": [
        "widowx_stack_cube",
        "widowx_put_eggplant_in_basket",
        "widowx_carrot_on_plate",
        "widowx_spoon_on_towel"
    ],
    "simpler_ood": [
        "widowx_redbull_on_plate",
        "widowx_zucchini_on_towel"
    ],
    # Individual task suites
    "simpler_stack_cube": ["widowx_stack_cube"],
    "simpler_put_eggplant_in_basket": ["widowx_put_eggplant_in_basket"],
    "simpler_spoon_on_towel": ["widowx_spoon_on_towel"],
    "simpler_carrot_on_plate": ["widowx_carrot_on_plate"],
    "simpler_redbull_on_plate": ["widowx_redbull_on_plate"],
    "simpler_carrot_on_plate_unseen_lighting": ["widowx_carrot_on_plate_unseen_lighting"],
    "simpler_tennis_ball_in_basket": ["widowx_tennis_ball_in_basket"],
    "simpler_toy_dinosaur_on_towel": ["widowx_toy_dinosaur_on_towel"],
    "simpler_zucchini_on_towel": ["widowx_zucchini_on_towel"],
    # Google Robot tasks (from SimplerEnv but not in RoboMonkey benchmark)
    "google_robot_basic": [
        "google_robot_pick_coke_can",
        "google_robot_move_near",
        "google_robot_open_drawer",
        "google_robot_close_drawer"
    ],
    "google_robot_pick": [
        "google_robot_pick_coke_can",
        "google_robot_pick_horizontal_coke_can",
        "google_robot_pick_vertical_coke_can",
        "google_robot_pick_standing_coke_can",
        "google_robot_pick_object"
    ],
    "google_robot_drawer": [
        "google_robot_open_drawer",
        "google_robot_open_top_drawer",
        "google_robot_open_middle_drawer",
        "google_robot_open_bottom_drawer",
        "google_robot_close_drawer",
        "google_robot_close_top_drawer",
        "google_robot_close_middle_drawer",
        "google_robot_close_bottom_drawer"
    ],
    "google_robot_place": [
        "google_robot_place_in_closed_drawer",
        "google_robot_place_in_closed_top_drawer",
        "google_robot_place_in_closed_middle_drawer",
        "google_robot_place_in_closed_bottom_drawer",
        "google_robot_place_apple_in_closed_top_drawer"
    ]
}

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # SimplerEnv environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "simpler_widowx"  # Task suite name
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20  # Number of rollouts per task

    #################################################################################################################
    # Evaluation parameters
    #################################################################################################################
    max_episode_length: int = 200  # Maximum episode length
    recording: bool = False  # Whether to record videos
    video_dir: str = "videos"  # Directory to save videos
    save_trajectories: bool = False  # Whether to save trajectory data
    trajectory_dir: str = "trajectories"  # Directory to save trajectories

    #################################################################################################################
    # Model parameters
    #################################################################################################################
    model_name: str = "pi05_droid"  # Model to use for evaluation
    checkpoint_dir: Optional[str] = None  # Override checkpoint directory

    #################################################################################################################
    # Logging and debugging
    #################################################################################################################
    log_level: str = "INFO"
    debug: bool = False


def main(args: Args):
    """Main evaluation function for SimplerEnv tasks."""
    
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create output directories
    if args.recording:
        os.makedirs(args.video_dir, exist_ok=True)
    if args.save_trajectories:
        os.makedirs(args.trajectory_dir, exist_ok=True)
    
    # Get task list
    if args.task_suite_name not in SIMPLER_TASK_SUITES:
        available_suites = list(SIMPLER_TASK_SUITES.keys())
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. Available: {available_suites}")
    
    task_list = SIMPLER_TASK_SUITES[args.task_suite_name]
    logger.info(f"Evaluating on task suite: {args.task_suite_name}")
    logger.info(f"Tasks: {task_list}")
    
    # Initialize policy client
    policy_client = _websocket_client_policy.WebSocketClientPolicy(
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )
    
    # Results storage
    all_results = {}
    
    # Evaluate each task
    for task_name in task_list:
        logger.info(f"Evaluating task: {task_name}")
        
        try:
            task_results = evaluate_task(
                task_name=task_name,
                policy_client=policy_client,
                args=args
            )
            all_results[task_name] = task_results
            
            # Log results
            success_rate = task_results["success_rate"]
            logger.info(f"Task {task_name} - Success rate: {success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Error evaluating task {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}
    
    # Save results
    results_file = f"simpler_results_{args.task_suite_name}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("SIMPLER ENV EVALUATION RESULTS")
    print("="*50)
    
    total_successes = 0
    total_episodes = 0
    
    for task_name, results in all_results.items():
        if "error" in results:
            print(f"{task_name}: ERROR - {results['error']}")
        else:
            success_rate = results["success_rate"]
            successes = results["successes"]
            episodes = results["episodes"]
            print(f"{task_name}: {success_rate:.2%} ({successes}/{episodes})")
            total_successes += successes
            total_episodes += episodes
    
    if total_episodes > 0:
        overall_success_rate = total_successes / total_episodes
        print(f"\nOverall Success Rate: {overall_success_rate:.2%} ({total_successes}/{total_episodes})")
    
    print(f"\nDetailed results saved to: {results_file}")


def evaluate_task(
    task_name: str,
    policy_client: _websocket_client_policy.WebSocketClientPolicy,
    args: Args
) -> dict:
    """Evaluate a single SimplerEnv task."""
    
    # Initialize environment
    env = simpler_env.make(task_name)
    
    # Results tracking
    successes = 0
    episodes = 0
    episode_results = []
    
    # Video recording setup
    video_writer = None
    if args.recording:
        video_path = os.path.join(args.video_dir, f"{task_name}_episode_{episodes}.mp4")
        video_writer = imageio.get_writer(video_path, fps=30)
    
    # Trajectory storage
    trajectory_data = []
    
    # Run episodes
    for episode in range(args.num_trials_per_task):
        logger.info(f"Running episode {episode + 1}/{args.num_trials_per_task}")
        
        # Reset environment
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        
        logger.info(f"Episode {episode + 1} - Instruction: {instruction}")
        
        # Episode tracking
        episode_data = {
            "instruction": instruction,
            "reset_info": reset_info,
            "observations": [],
            "actions": [],
            "rewards": [],
            "success": False
        }
        
        # Action planning with receding horizon
        action_plan = collections.deque()
        
        # Run episode
        for step in range(args.max_episode_length):
            # Get current image (following RoboMonkey processing)
            img = np.ascontiguousarray(get_image_from_maniskill2_obs_dict(env, obs))
            
            # Apply RoboMonkey's image preprocessing (JPEG encode/decode + resize)
            # Encode as JPEG, as done in RLDS dataset builder
            img = tf.image.encode_jpeg(img)
            img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
            # Resize to base size then to final size (following RoboMonkey)
            img = tf.image.resize(img, (128, 128), method="lanczos3", antialias=True)
            img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
            img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
            img = img.numpy()
            
            # Record observation
            episode_data["observations"].append({
                "step": step,
                "image_shape": img.shape,
                "state": obs.copy() if isinstance(obs, dict) else obs
            })
            
            # Record video frame
            if video_writer is not None:
                video_writer.append_data(img)
            
            # Get action if plan is empty
            if not action_plan:
                # Prepare observation for policy (matching OpenPI format)
                # Extract state from SimplerEnv observation (following RoboMonkey format)
                state = obs.get("extra", {}).get("tcp_pose", np.zeros(7))  # RoboMonkey uses tcp_pose
                
                observation = {
                    "observation/image": img,
                    "observation/wrist_image": np.zeros_like(img),  # SimplerEnv doesn't use wrist images
                    "observation/state": state,
                    "prompt": instruction
                }
                
                # Query policy
                try:
                    action_chunk = policy_client.infer(observation)["actions"]
                    action_plan.extend(action_chunk[:args.replan_steps])
                except Exception as e:
                    logger.error(f"Policy inference failed: {e}")
                    action_plan.append(np.zeros(7))  # 7D default action for SimplerEnv
            
            # Execute action
            if action_plan:
                action = action_plan.popleft()
            else:
                action = np.zeros(7)  # 7D default action for SimplerEnv
            
            # Record action
            episode_data["actions"].append(action.tolist())
            
            # Step environment
            obs, reward, success, truncated, info = env.step(action.copy())
            
            # Record reward
            episode_data["rewards"].append(reward)
            
            # Check if episode is done
            if success or truncated:
                episode_data["success"] = success
                episode_data["truncated"] = truncated
                episode_data["final_info"] = info
                
                if success:
                    successes += 1
                    logger.info(f"Episode {episode + 1} - SUCCESS!")
                else:
                    logger.info(f"Episode {episode + 1} - Failed")
                
                break
        
        # Store episode data
        trajectory_data.append(episode_data)
        episodes += 1
        
        # Close video writer if recording
        if video_writer is not None:
            video_writer.close()
            if episode_data["success"]:
                # Rename successful videos
                success_video_path = os.path.join(args.video_dir, f"{task_name}_episode_{episodes-1}_success.mp4")
                os.rename(video_path, success_video_path)
            
            # Setup for next episode
            if episode < args.num_trials_per_task - 1:
                video_path = os.path.join(args.video_dir, f"{task_name}_episode_{episodes}.mp4")
                video_writer = imageio.get_writer(video_path, fps=30)
    
    # Calculate results
    success_rate = successes / episodes if episodes > 0 else 0.0
    
    results = {
        "success_rate": success_rate,
        "successes": successes,
        "episodes": episodes,
        "task_name": task_name
    }
    
    # Save trajectory data if requested
    if args.save_trajectories:
        trajectory_file = os.path.join(args.trajectory_dir, f"{task_name}_trajectories.pkl")
        with open(trajectory_file, "wb") as f:
            pickle.dump(trajectory_data, f)
        logger.info(f"Saved trajectories to: {trajectory_file}")
    
    # Close environment
    env.close()
    
    return results


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
