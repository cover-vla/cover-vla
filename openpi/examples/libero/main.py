import collections
import dataclasses
import logging
import math
import pathlib
import json
import copy
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import sys
import os
import pickle

sys.path.append("../clip_verifier/scripts")
from vla_dino_inference import VLA_DINO_Inference

import torch

torch.set_num_threads(8)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

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
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 20  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # Language transformation
    #################################################################################################################
    lang_transform_type: str = "no_transform"  # 'rephrase' or 'no_transform'
    num_rephrase_candidates: int = 3       # number of rephrased instructions to use (not counting original)
    
    # --- Load rephrases JSON and initialize verifier ---
    rephrases_json_path: str = '../openvla/experiments/robot/libero/libero_rephrase_hard.json'
    vla_clip_model_path: str = '../clip_verifier/bash/trajectory_checkpoints/libero_spatial_oft_all_final_best.pt'
    history_length: int = 10
    score_threshold: float = 0.5


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    with open(args.rephrases_json_path, 'r') as f:
        all_rephrases = json.load(f)

    verifier = VLA_DINO_Inference(
        model_path=args.vla_clip_model_path,
        history_length=args.history_length,
        use_transformer=True
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)[5:]):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # --- Load rephrases for this task ---
        rephrased_list = all_rephrases[args.task_suite_name][str(task_id)]["rephrases"]
        if args.lang_transform_type == "no_transform":
            candidate_instructions = [task_description]
        else:
            candidate_instructions = rephrased_list[:args.num_rephrase_candidates]
            
        # if len(candidate_instructions) > 3:
        #     candidate_instructions[3] = task_description + "."

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            episode_scores = []
            episode_actions = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # --- Always generate and score new action chunks for all candidates ---
                # Get preprocessed image
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(img)

                candidate_action_chunks = []
                for instr in candidate_instructions:
                    element = {
                        "observation/image": copy.deepcopy(img),
                        "observation/wrist_image": copy.deepcopy(wrist_img),
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(instr),
                        # "prompt": "Open the drawer.",
                    }
                    action_chunk = client.infer(element)["actions"]
                    print ("length of action chunk", len(action_chunk))
                    input()
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    candidate_action_chunks.append(action_chunk)
                # --- Batch score all candidates ---
                _, scores_dict = verifier.predict(
                    (img, wrist_img),
                    candidate_instructions,
                    process_action(copy.deepcopy(candidate_action_chunks))
                )
                scores = [scores_dict[str(i)] for i in range(len(candidate_action_chunks))]
                best_idx = int(np.argmax(scores))
                best_score = scores[best_idx]
                best_actions = candidate_action_chunks[best_idx]
                best_instr = candidate_instructions[best_idx]
                print(f"[t={t}] Selected instruction: '{best_instr}' (Score: {best_score:.3f})")
                action = best_actions[0]

                # Save action and score for this step
                episode_actions.append(action.tolist() if hasattr(action, 'tolist') else action)
                episode_scores.append(best_score)

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1


            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                transform_type=args.lang_transform_type,
                task_description=task_description,
                score_list=episode_scores,
                action_list=episode_actions,
                clip_update_num=args.num_rephrase_candidates,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def process_action(action_chunk_list):
    processed_action_chunk_list = []
    for action_chunk in action_chunk_list:
        processed_action_chunk = []
        for action in action_chunk:
            action[6] = 1.0 if action[6] > 0.0 else -1.0
            processed_action_chunk.append(action)
        processed_action_chunk_list.append(np.array(processed_action_chunk))
    return processed_action_chunk_list

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(rollout_images, idx, success, transform_type,
                       task_description, score_list=None, 
                       action_list=None, clip_update_num=None):
    """Saves an MP4 replay of an episode and a .pkl file with actions and scores."""
    rollout_dir = f"./rollouts/{transform_type}_{clip_update_num}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")

    mean_score = np.nanmean(score_list) if score_list else None
    if mean_score is not None and not np.isnan(mean_score):
        score_str = f"{mean_score:.3f}"
    else:
        score_str = "None"

    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--score={score_str}--task={processed_task_description}.mp4"
    data_path = f"{rollout_dir}/episode={idx}--success={success}--score={score_str}--task={processed_task_description}.pkl"
    import imageio
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if score_list is not None and action_list is not None:
        data = {
            "score_list": score_list,
            "action_list": action_list,
        }
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved data at path {data_path}")
    return mp4_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
