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
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Dict
from collections import deque
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
    process_raw_image_to_jpg,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import imageio
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
import time
import random
from sentence_transformers import SentenceTransformer

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

def save_rollout_video_openpi(rollout_images, idx, success, task_description, transformation_type, model_name, lang_rephrase_num, policy_batch_inference_size, log_file=None):
    """Saves an MP4 replay of an episode."""
    if model_name == "juexzz/INTACT-pi0-finetune-rephrase-bridge":
        rollout_dir = f"./test_rollouts_openpi_rephrase_selection/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    elif model_name == "juexzz/INTACT-pi0-finetune-bridge":
        rollout_dir = f"./test_rollouts_openpi_original_selection/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
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
        rollout_dir = f"./test_rollouts_openpi_rephrase_selection/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
    elif model_name == "juexzz/INTACT-pi0-finetune-bridge":
        rollout_dir = f"./test_rollouts_openpi_original_selection/transform_{transformation_type}/lang_{lang_rephrase_num}_sample_{policy_batch_inference_size}"
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

# import sys
sys.path.append('/root/vla-clip/lerobot_intact')
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import torch

# Import ensemble model for similarity scoring
sys.path.append('/root/vla-clip/bridge_verifier/ensemble_eval')
from efficient_ensemble_merged import EfficientEnsembleMerged

def load_rephrases(task_suite_name: str):
    """Load pre-generated rephrases for the task suite."""
    # Make the path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'simpler_rephrased.json')
    
    with open(json_path, 'r') as f:
        all_rephrases = json.load(f)
    # The new format has an "instructions" key containing the task data
    return all_rephrases.get("instructions", {})

class QwenEmbedder:
    """Qwen3-based text embedder for computing instruction embeddings."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "auto"):
        """Initialize Qwen3 embedder."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Initialized Qwen3 embedder: {model_name} on {device}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute Qwen3 embeddings for a list of texts using query prompt."""
        embeddings = self.model.encode(
            texts, 
            prompt_name="query",
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings

def compute_rephrase_embeddings(rephrases_data: Dict, num_clusters: int, embedder: QwenEmbedder):
    """
    Compute Qwen3 embeddings for rephrases and cluster them into specified number of clusters.
    Returns clustered rephrases for each task.
    
    Args:
        rephrases_data: Dictionary from load_rephrases() containing task rephrases
        num_clusters: Number of clusters to create (e.g., 4 or 8)
    
    Returns:
        Dictionary mapping task_key -> List[str] of clustered rephrases (one per cluster)
    """
    from sklearn.cluster import KMeans
    
    print(f"Computing Qwen3 embeddings and clustering into {num_clusters} clusters...")
    
    # Compute embeddings for all rephrases
    embeddings = embedder.embed_texts(rephrases_data, batch_size=32)
    
    # Cluster embeddings using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Select one representative rephrase from each cluster (closest to cluster center)
    clustered_rephrases = []
    
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        # Find rephrase closest to cluster center
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        clustered_rephrases.append(rephrases_data[closest_idx])
    
    print(f"Clustering complete! Processed {len(clustered_rephrases)} clusters.")
    return clustered_rephrases

def process_inputs(batch_size, predefined_action_queue, verifier_action=False, action_history=[],cfg=None):
    processed_future_actions_batch = []
    for i in range(cfg.n_action_steps):
        # Get single action (batch_size, 7) and process it
        single_action = predefined_action_queue[i].cpu().numpy()  # Shape: (batch_size, 7)
        processed_execution_actions_for_step = []
        for batch_idx in range(batch_size):
            sample_1x7 = single_action[batch_idx:batch_idx+1]          # (1,7)
            processed_execution_1x7 = convert_maniskill_with_bridge_adapter(sample_1x7, verifier_action=verifier_action, action_ensemble_temp=-0.8)
            # print ("processed_1x7:", processed_1x7)
            # ensure (1,7) ndarray
            processed_execution_1x7 = np.asarray(processed_execution_1x7)
            processed_execution_actions_for_step.append(processed_execution_1x7)

        # stack to (batch,1,7) then squeeze to (batch,7)
        processed_batch = np.vstack(processed_execution_actions_for_step)        # (batch, 1, 7) if adapter returns (1,7)
        processed_batch = processed_batch.reshape(batch_size, 7)       # -> (batch,7)
        processed_future_actions_batch.append(processed_batch)
        
    num_past = min(len(action_history), 6)
    future_actions = np.stack(processed_future_actions_batch)  # (cfg.n_action_steps, batch_size, action_dim)
    future_actions_transposed = future_actions.transpose(1, 0, 2)  # (batch_size, cfg.n_action_steps, action_dim)
    if num_past > 0:
        past_actions = np.stack(action_history[-num_past:])
        past_actions = np.expand_dims(past_actions, axis=0).repeat(batch_size, axis=0) # (cfg.batch_inference_size, num_past, action_dim)
        # Concatenate along timestep dimension
        processed_full_trajectory = np.concatenate([past_actions, future_actions_transposed], axis=1)  # (cfg.batch_inference_size, num_past+cfg.n_action_steps, action_dim)
    else:
        processed_full_trajectory = future_actions_transposed
    action_histories_list = [processed_full_trajectory[i] for i in range(batch_size)]
        
    return action_histories_list


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
    # Robomonkey Config
    initial_samples: int = 5
    augmented_samples: int = 32
    # Action chunking (match INT-ACT style multi-step actions)
    n_action_steps: int = 4
    
    # language selection parameters 
    language_index: int = 0
    
    # Language transformation parameters
    lang_transform_type: str = "no_transform"            # Type of language transformation (rephrase/no_transform)
    lang_rephrase_num: int = 8
    # Batch inference parameters
    policy_batch_inference_size: int = 2                        # Number of samples for batch inference (same instruction repeated)
    
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
    if cfg.use_verifier:
        # Initialize ensemble model for similarity scoring
        print("Loading ensemble model for similarity scoring...")
        ensemble_model = EfficientEnsembleMerged("/root/vla-clip/bridge_verifier/ensemble_789_trainable_only.pt")
        print("Ensemble model loaded successfully!")
    else:
        ensemble_model = None
    # Initialize SIMPLER task suite

    task_suite = get_benchmark(cfg.task_suite_name)()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    
    # Load pre-generated rephrases if available
    preloaded_rephrases = load_rephrases(cfg.task_suite_name)
    
    action_queue = None
    # Create adapter for preprocessing (singleton pattern)
    if not hasattr(pi0_policy, '_preprocess_adapter'):
        pi0_policy._preprocess_adapter = create_bridge_adapter_wrapper(cfg.action_ensemble_temp)
    preprocess_adapter = pi0_policy._preprocess_adapter
    
    # use pre-defined action noise std 
    action_noise_std = 1.0 if cfg.policy_batch_inference_size > 1 else 1.0
    
    # create embedder
    embedder = QwenEmbedder()

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
            # Find matching task in preloaded rephrases
            matching_task_id = None
            for task_key, task_data in preloaded_rephrases.items():
                if task_key == original_task_description:
                    matching_task_id = task_key
                    break
            
            if matching_task_id is not None:
                # rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"]
                # Cluster rephrases if using rephrase transformation
                if cfg.lang_rephrase_num > 1:
                    print(f"Clustering rephrases into {cfg.lang_rephrase_num} clusters...")
                    rephrased_list = compute_rephrase_embeddings(preloaded_rephrases[matching_task_id]["ert_rephrases"], num_clusters=cfg.lang_rephrase_num, embedder=embedder)
                else:
                    rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"][:1] 
            else:
                raise ValueError(f"No preloaded rephrases found for task: {original_task_description}")

        if cfg.lang_transform_type == "no_transform":
            assert cfg.lang_rephrase_num == 1, "Language rephrase number must be 1 for no transformation"
        # Start episodes
        task_episodes, task_successes = 0, 0
        for trail_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if matching_task_id is not None:
                task_description = preloaded_rephrases[matching_task_id]["ert_rephrases"][cfg.language_index]
            else:
                task_description = original_task_description
            # print(f"Using task description: {task_description}")
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            if trail_idx % 50 == 0:
                seeds = itertools.count(1000)
            # Reset environment to specified seed (initial state)
            obs, reset_info = env.reset(seed=next(seeds))

            # Setup
            t = 0
            replay_images = []
            action_history = []  # Track actions for similarity scoring
            # action_queue = None  # Deque of length n_action_steps; regenerate only when empty
            
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
                batch_size = cfg.policy_batch_inference_size * cfg.lang_rephrase_num
                
                # 1) Build the unique instruction list
                if rephrased_list is not None and cfg.lang_rephrase_num > 1:
                    unique_prompts = [task_description] + rephrased_list[: cfg.lang_rephrase_num - 1]
                else:
                    # no rephrases -> just the original
                    unique_prompts = [task_description]
                    # (optional) keep configs consistent
                    # assert cfg.lang_rephrase_num == 1
                # unique_prompts = rephrased_list[:cfg.lang_rephrase_num]

                # 2) Repeat each instruction cfg.policy_batch_inference_size times
                task_list = []
                for p in unique_prompts:
                    task_list.extend([p] * cfg.policy_batch_inference_size)
                    
                assert len(task_list) == batch_size, "Batch size mismatch"
                    
                # Create batch observation dict for LeRobot PI0
                # Replicate image and state to match batch size
                batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
                batch_state = processed_obs['observation.state'].repeat(batch_size, 1)
                
                observation = {
                    image_key: batch_image,  # [batch_size, 3, 224, 224]
                    "observation.state": batch_state,  # [batch_size, 7]
                    "task": task_list,  # List instructions
                }
                # Only call select_action every n_action_steps to avoid interfering with policy's internal queue management
                if t % cfg.n_action_steps == 0:
                    with torch.no_grad():
                        output_action_queue = pi0_policy.select_action(observation, noise_std=action_noise_std)
                        action_queue = output_action_queue.copy()
                        output_action_queue.clear()
                if cfg.use_verifier and t % cfg.n_action_steps == 0:
                    assert len(action_queue) == cfg.n_action_steps, f"Action queue length should be {cfg.n_action_steps}, but got {len(action_queue)}"
                    num_past = min(len(action_history), 6)
                    # Create a proper copy of the action_queue by converting to list and back to deque
                    predefined_action_queue = list(action_queue)
                    action_queue.popleft()
                    action_histories_list = process_inputs(batch_size, predefined_action_queue, verifier_action=True, action_history=action_history.copy(), cfg=cfg)
                    # print ("action_histories_list:", action_histories_list)
                    images_list = [process_raw_image_to_jpg(raw_img)] * batch_size
                    # print ("image_list shape:", np.array(images_list).shape)
                    # print ("action_histories_list shape:", np.array(action_histories_list).shape)
                    # print ("task_description shape:", np.array([task_description] * cfg.policy_batch_inference_size).shape)
                    # print ("cfg.n_action_steps:", cfg.n_action_steps)
                    with torch.no_grad():
                        max_score, max_instruction, max_action_history, global_action_idx = ensemble_model.compute_max_similarity_scores_batch(
                            images=images_list[0:cfg.policy_batch_inference_size],
                            instructions=[task_description] * cfg.policy_batch_inference_size,
                            all_action_histories=action_histories_list[0:cfg.policy_batch_inference_size],
                            cfg_repeat_language_instructions=cfg.policy_batch_inference_size
                        )
                    if max_score < 0.1:
                        with torch.no_grad():
                            max_score, max_instruction, max_action_history, global_action_idx = ensemble_model.compute_max_similarity_scores_batch(
                                images=images_list,
                                instructions=task_list,
                                all_action_histories=action_histories_list,
                                cfg_repeat_language_instructions=cfg.policy_batch_inference_size
                            )
                    # Select action from execution-format actions, not verification-format
                    # The verifier gives us the best trajectory index, but we need to get the corresponding execution action from processed execution actions
                    execution_action_histories_list = process_inputs(batch_size, predefined_action_queue, verifier_action=False, action_history=action_history.copy(), cfg=cfg)
                    execute_action = execution_action_histories_list[global_action_idx][num_past].copy()  # shape (7,)
                    # print ("selected action:", execution_action_histories_list[global_action_idx])
                    # Get the group of actions corresponding to the selected global_action_idx
                    group_start = (global_action_idx // cfg.policy_batch_inference_size) * cfg.policy_batch_inference_size
                    group_end = group_start + cfg.policy_batch_inference_size
                    stacked_histories = np.stack(execution_action_histories_list[group_start:group_end])  # (batch_size, num_past + n_action_steps, 7)
                    # print ("stacked_histories:", stacked_histories.shape)
                    # print ("stacked_histories:", stacked_histories)
                    grippers = stacked_histories[:, num_past, -1]  # (batch_size,) - extract gripper values at timestep num_past
                    # print ("grippers:", grippers)
                    # Count votes: >= 0 is +1 (closed), < 0 is -1 (open)
                    close_votes = int((grippers >= 0).sum())
                    open_votes  = int((grippers < 0).sum())

                    if close_votes > open_votes:
                        execute_action[-1] = 1.0   # Open gripper
                    elif open_votes > close_votes:
                        execute_action[-1] = -1.0  # Close gripper
                    else:
                        # Tie → default to the verifier's selected action's sign
                        execute_action[-1] = 1.0 if execute_action[-1] >= 0 else -1.0

                    # Ensure exactly -1 or 1
                    execute_action[-1] = float(np.sign(execute_action[-1]))
                    
                    # Extract remaining actions from the selected batch item (global_action_idx) from the original policy output
                    # predefined_action_queue still contains all n_action_steps timesteps (before popleft)
                    # We executed action from timestep 0, batch item global_action_idx
                    # Now extract remaining actions from timesteps 1, 2, 3 for batch item global_action_idx
                    selected_action_chunk = deque()
                    for timestep_idx in range(1, cfg.n_action_steps):
                        # Get the action tensor for this timestep (shape: batch_size, 7)
                        timestep_actions = predefined_action_queue[timestep_idx]
                        # Extract the action for the selected batch item
                        selected_action = timestep_actions[global_action_idx:global_action_idx+1]  # shape (1, 7)
                        selected_action_chunk.append(selected_action)
                    
                    # Replace action_queue with the selected chunk actions (now contains only actions for selected batch item)
                    action_queue = selected_action_chunk
                    
                    # Store verifier data
                    episode_data['verifier_scores'].append(max_score)
                    episode_data['selected_instructions'].append(max_instruction)
                    episode_data['execute_actions'].append(execute_action.copy())
                    episode_data['step_timestamps'].append(t)
                    
                    task_description = max_instruction
                else:
                    # Use actions from queue (either from verifier-selected chunk or from regular policy)
                    # action_queue should exist from previous call to select_action at n_action_steps boundary
                    single_action = action_queue.popleft().cpu().numpy()
                    action_for_env = single_action[0:1]
                    execute_action = convert_maniskill_with_bridge_adapter(action_for_env, verifier_action=False, action_ensemble_temp=cfg.action_ensemble_temp)
                    # Store data for non-verifier case
                    # print ("execute_action no verifier:", execute_action)
                    episode_data['verifier_scores'].append(None)
                    episode_data['selected_instructions'].append(task_description)
                    episode_data['execute_actions'].append(execute_action.copy())
                    episode_data['step_timestamps'].append(t)
                    
                # If verifier is used, also add action to history with verifier_action=True
                if cfg.use_verifier:
                    if t % cfg.n_action_steps == 0:
                        processed_action_for_history = max_action_history[num_past].copy()
                    else:
                        processed_action_for_history = convert_maniskill_with_bridge_adapter(
                            single_action[0:1], verifier_action=True, action_ensemble_temp=cfg.action_ensemble_temp
                        )
                        # print ("processed_action_for_history:", processed_action_for_history)
                        # input("processed_action_for_history")
                    action_history.append(processed_action_for_history)

                # print ("execute_action:", execute_action)
                # print ("insturction:", task_description)
                # input()
                # print ("execute_action:", execute_action)
                obs, reward, done, trunc, info = env.step(execute_action)    
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                
                # Update progress bar with similarity score if verifier is used
                if cfg.use_verifier:
                    pbar.set_description(f"Episode steps (score: {max_score:.3f})")
                pbar.update(1)


            pbar.close()
            task_episodes += 1
            total_episodes += 1

            # Update episode data with final information
            episode_data['success'] = done
            episode_data['episode_length'] = t

            action_queue.clear()
            # Save a replay video of the episode
            save_rollout_video_openpi(
                replay_images, total_episodes, success=done, 
                task_description=task_description, 
                transformation_type=cfg.lang_transform_type,
                model_name=cfg.pretrained_checkpoint,
                lang_rephrase_num=cfg.lang_rephrase_num,
                policy_batch_inference_size=cfg.policy_batch_inference_size,
                log_file=log_file
            )
            
            # Save episode data to pickle file
            save_episode_data_openpi(
                episode_data, total_episodes, success=done, 
                task_description=task_description, 
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
