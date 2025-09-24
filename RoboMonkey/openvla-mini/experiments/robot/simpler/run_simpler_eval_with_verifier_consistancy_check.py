
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Dict
import requests
import json_numpy as json
import clip
import torch

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
    get_gaussian_vla_action,
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
    batch_server_url: str = "http://localhost:3200"  # URL of the SGLang batch server
    batch_temperature: float = 0                   # Temperature for batch inference
    save_image_name: str = "reward_img.jpg"

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
    
    repeated_samples: int = 1

    #################################################################################################################
    # Consistency Check Parameters
    #################################################################################################################
    use_consistency_check: bool = True               # Enable consistency check and consistency-based softmax temperature
    max_reasonable_action_distance: float = 1.1      # Maximum reasonable action distance for similar embeddings
    consistency_top_k: int = 5                       # Number of top-ranked actions to check for consistency
    base_softmax_temperature: float = 0.01            # Base temperature for softmax sampling
    consistency_temperature_scale: float = 3.0       # Scale factor for consistency-based temperature adjustment


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
    json_path = os.path.join(script_dir, 'simpler_rephrased_final_eval.json')
    
    try:
        with open(json_path, 'r') as f:
            all_rephrases = json.load(f)
        # The new format has an "instructions" key containing the task data
        return all_rephrases.get("instructions", {})
    except FileNotFoundError:
        print(f"Warning: Could not find rephrase file {json_path}. Using empty rephrases.")
        return {}


def compute_embedding_distance(model, instruction1: str, instruction2: str) -> float:
    """
    Compute embedding distance between two instructions using CLIP text encoder.
    This measures the semantic similarity between instructions.
    """
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Tokenize text
    text1 = clip.tokenize([instruction1.lower()]).to(device)
    text2 = clip.tokenize([instruction2.lower()]).to(device)
    
    # Get embeddings
    with torch.no_grad():
        emb1 = model.encode_text(text1)[0]  # Get first (and only) embedding
        emb2 = model.encode_text(text2)[0]  # Get first (and only) embedding
    
    # Normalize embeddings
    emb1_norm = emb1 / torch.linalg.norm(emb1)
    emb2_norm = emb2 / torch.linalg.norm(emb2)
    cosine_similarity = torch.cosine_similarity(emb1_norm, emb2_norm, dim=0)
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance.cpu().numpy()


def check_action_consistency_with_embeddings(model, candidate_instructions: List[str], 
                                            candidate_actions: List[np.ndarray],
                                            cfg: GenerateConfig,
                                            embeddings_cache: Optional[Dict[str, np.ndarray]] = None) -> tuple[float, dict]:
    """
    Check consistency using Lipschitz continuity: similar embeddings should produce similar actions.
    
    For Lipschitz continuity, we want: ||f(x1) - f(x2)|| <= L * ||x1 - x2||
    Where f is the embedding->action mapping, and L is the Lipschitz constant.
    
    Args:
        candidate_instructions: List of top-K candidate instructions
        candidate_actions: List of corresponding actions
        cfg: Configuration object
        
    Returns:
        Tuple of (consistency_score, consistency_info)
    """
    if len(candidate_instructions) < 2:
        return 1.0, {"embedding_distances": [], "action_distances": [], "lipschitz_ratios": [], "lipschitz_constant": 0.0}
    
    # Calculate pairwise embedding distances and action distances
    embedding_distances = []
    action_distances = []
    lipschitz_ratios = []
    
    for i in range(len(candidate_instructions)):
        for j in range(i + 1, len(candidate_instructions)):
            # Compute embedding distance (cosine distance)
            if embeddings_cache is not None:
                emb_i = embeddings_cache[candidate_instructions[i]]
                emb_j = embeddings_cache[candidate_instructions[j]]
                # embeddings are L2-normalized; cosine distance = 1 - dot
                emb_dist = float(1.0 - np.dot(emb_i, emb_j))
            else:
                emb_dist = compute_embedding_distance(
                    model,
                    candidate_instructions[i], 
                    candidate_instructions[j]
                )
            embedding_distances.append(emb_dist)
            
            # Compute action distance (L2 norm)
            action_dist = np.linalg.norm(
                np.array(candidate_actions[i]) - np.array(candidate_actions[j])
            )
            action_distances.append(action_dist)
            
            # Compute Lipschitz ratio: action_distance / embedding_distance
            # If embedding_distance is very small, skip to avoid division by zero
            if emb_dist > 1e-6:  # Avoid division by very small numbers
                lipschitz_ratio = action_dist / emb_dist
                lipschitz_ratios.append(lipschitz_ratio)
    
    # Convert to numpy arrays
    embedding_distances = np.array(embedding_distances)
    action_distances = np.array(action_distances)
    lipschitz_ratios = np.array(lipschitz_ratios)
    
    # Compute Lipschitz constant (maximum ratio)
    if len(lipschitz_ratios) > 0:
        lipschitz_constant = np.max(lipschitz_ratios)
        mean_lipschitz_ratio = np.mean(lipschitz_ratios)
        std_lipschitz_ratio = np.std(lipschitz_ratios)
    else:
        lipschitz_constant = 0.0
        mean_lipschitz_ratio = 0.0
        std_lipschitz_ratio = 0.0
    
    # Compute correlation for additional insight
    if len(embedding_distances) > 1 and np.std(embedding_distances) > 0 and np.std(action_distances) > 0:
        correlation = np.corrcoef(embedding_distances, action_distances)[0, 1]
    else:
        correlation = 1.0  # Perfect correlation if no variance
    
    # Consistency score based on Lipschitz continuity
    # Lower Lipschitz constant = better consistency (more bounded)
    # We normalize by a reasonable threshold
    max_reasonable_lipschitz = cfg.max_reasonable_action_distance / 0.5  # minimum embedding distance is 0.5
    normalized_lipschitz = min(lipschitz_constant / max_reasonable_lipschitz, 1.0)
    consistency_score = max(0.0, 1.0 - normalized_lipschitz)
    
    consistency_info = {
        "embedding_distances": embedding_distances.tolist(),
        "action_distances": action_distances.tolist(),
        "lipschitz_ratios": lipschitz_ratios.tolist(),
        "lipschitz_constant": lipschitz_constant,
        "mean_lipschitz_ratio": mean_lipschitz_ratio,
        "std_lipschitz_ratio": std_lipschitz_ratio,
        "correlation": correlation,
        "consistency_score": consistency_score
    }
    
    return consistency_score, consistency_info


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

    if cfg.use_consistency_check:
        print(f"Using Consistency Check: max_action_dist={cfg.max_reasonable_action_distance}, top_k={cfg.consistency_top_k}")
        print(f"  Base temperature: {cfg.base_softmax_temperature}, Scale: {cfg.consistency_temperature_scale}")

    else:
        print("Consistency Check DISABLED.")

    # if cfg.repeated_samples <= 1:
    #     assert cfg.batch_temperature == 0, "Batch temperature must be 0 if repeated samples is 1."
    # else:
    #     assert cfg.batch_temperature > 0, "Batch temperature must be greater than 0 if repeated samples is greater than 1."
    
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    else:
        cfg.unnorm_key = "bridge_orig"
        
    batch_temperature = cfg.batch_temperature
    repeated_samples = cfg.repeated_samples

    # Load model
    # model = get_model(cfg)
    # Load CLIP model for text embeddings
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
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

        # Build per-task CLIP text embedding cache for the original and all rephrases
        task_instruction_to_embedding: Dict[str, np.ndarray] = {}
        task_all_instructions: List[str] = []
        # Determine rephrases for this specific task description
        matching_task_id = None
        for task_key, task_data in preloaded_rephrases.items():
            if task_key == original_task_description:
                matching_task_id = task_key
                break
        if matching_task_id is not None:
            rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"]
            user_input_language_instruction = preloaded_rephrases[matching_task_id]["original"]
            task_all_instructions = [user_input_language_instruction] + list(rephrased_list)
        else:
            # Fallback to only the original if no rephrases found
            task_all_instructions = [original_task_description]

        # Precompute normalized embeddings once per task
        with torch.no_grad():
            device = next(clip_model.parameters()).device
            for instr in task_all_instructions:
                if instr not in task_instruction_to_embedding:
                    tokens = clip.tokenize([instr.lower()]).to(device)
                    emb = clip_model.encode_text(tokens)[0]
                    emb = emb / torch.linalg.norm(emb)
                    task_instruction_to_embedding[instr] = emb.detach().cpu().numpy()

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
            all_consistency_scores = []
            
            # if number of candidates is 1, set batch temperature to 0 and repeated samples to 1
            if cfg.clip_select_action_num_candidates == 1:
                batch_temperature = 0
                repeated_samples = 1

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
                    rephrased_list = preloaded_rephrases[matching_task_id]["ert_rephrases"]
                    user_input_language_instruction = preloaded_rephrases[matching_task_id]["original"]
                    candidate_instructions = [user_input_language_instruction] + rephrased_list[:cfg.clip_select_action_num_candidates-1]
                else:
                    print(f"No preloaded rephrases found for task: {original_task_description}")
                    raise ValueError(f"No preloaded rephrases found for task: {original_task_description}")
            log_file.write(f"Candidate instructions: {candidate_instructions}\n")
            
            if cfg.clip_select_action_num_candidates == 1:
                candidate_instructions = [user_input_language_instruction]
            
            while t < max_steps + cfg.num_steps_wait:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, trunc, info = env.step(get_simpler_dummy_action(cfg.model_family))
                    t += 1
                    pbar.update(1)
                    continue

                # Get preprocessed image
                get_simpler_img(env, obs, resize_size, save_image_name=cfg.save_image_name)
                img_verifier = get_simpler_img(env, obs, resize_size, verifier=True)
                # Save preprocessed image for replay video
                image_path = f"./transfer_images/{cfg.save_image_name}"
                replay_images.append(imageio.imread(image_path))

                # --- Action Generation and VLA-CLIP Scoring ---
                action_to_execute = None
                current_vla_clip_score = np.nan
                current_history_for_scoring = None
                consistency_score = np.nan

                batch_actions = []
                batch_input_instructions = []
                batch_instructions = []
                # use for the Scoring process
                for each_instruction in candidate_instructions:
                    batch_instructions.extend([each_instruction] * repeated_samples)
                
                # use for the gaussian sampling process    
                actual_samples = 5 if repeated_samples >= 5 else repeated_samples
                for each_instruction in candidate_instructions:
                    batch_input_instructions.extend([each_instruction] * actual_samples)
                actions = get_gaussian_vla_action(cfg, actual_samples, repeated_samples, batch_input_instructions, image_path, batch_temperature)
                batch_actions.extend(actions)
                
                # _, batch_actions = get_batch_actions(
                #     candidate_instructions, 
                #     image_path, 
                #     cfg.batch_server_url, 
                #     temperature=0
                # )

                if batch_actions is not None:
                    predicted_actions = [convert_maniskill(action) for action in batch_actions]
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
                        instructions = batch_instructions[0]
                    
                    # Use batch scoring for efficiency
                    scores = vla_clip_scorer.get_history_score(
                        img_verifier,
                        instructions,
                        padded_histories
                    )
                    scores = np.array(scores)
                else:
                    scores = np.ones(len(predicted_actions))  # Dummy scores if no scorer
                    padded_histories = [None] * len(predicted_actions)

                # --- Consistency Check and Combined Scoring ---
                consistency_info = {}
                if cfg.use_consistency_check and len(batch_instructions) > 1:
                    # Get top-K actions based on VLA-CLIP scores for consistency check
                    valid_indices = np.where(scores > -np.inf)[0]
                    if len(valid_indices) > 1:
                        # Get top-K actions based on scores
                        top_k = min(cfg.consistency_top_k, len(valid_indices))
                        top_indices = valid_indices[np.argsort(scores[valid_indices])[-top_k:]]
                        
                        # do a majority vote for last dimension of the top-k actions, save that last dimension value to overide the predict action last dimension value
                        # Prepare corresponding instructions and actions (no majority override here;
                        # majority vote is applied within get_gaussian_vla_action per instruction group)
                        top_instructions = [batch_instructions[idx] for idx in top_indices]
                        top_actions = [np.array(predicted_actions[idx]) for idx in top_indices]
                            
                        consistency_score, consistency_info = check_action_consistency_with_embeddings(
                            clip_model,
                            top_instructions,
                            top_actions,
                            cfg,
                            embeddings_cache=task_instruction_to_embedding
                        )
                    else:
                        print(f"  [t={t}] Not enough valid actions for consistency check (need >1, got {len(valid_indices)})")
                        log_file.write(f"  [t={t}] Not enough valid actions for consistency check (need >1, got {len(valid_indices)})\n")

                # --- Action Selection with Consistency-based Softmax Temperature ---
                if cfg.use_consistency_check and consistency_score is not None:
                    
                    # Calculate temperature based on consistency score (Lipschitz ratio)
                    # High consistency (close to 1) → low temperature (sharp distribution)
                    # Low consistency (close to 0) → high temperature (flat distribution)
                    temperature = cfg.base_softmax_temperature * (1.0 + cfg.consistency_temperature_scale * (1.0 - consistency_score))
                    
                    # log_file.write(f"  [t={t}] Consistency-based softmax: consistency_score={consistency_score:.3f}, temperature={temperature:.3f}\n")
                    
                    # Store temperature for use in softmax sampling
                    current_softmax_temperature = temperature
                    combined_scores = scores
                else:
                    # Use only VLA-CLIP scores
                    combined_scores = scores
                    current_softmax_temperature = cfg.base_softmax_temperature
                    print(f"  [t={t}] Using VLA-CLIP scoring only")
                    log_file.write(f"  [t={t}] Using VLA-CLIP scoring only\n")

                # Select the best action based on combined scores
                if cfg.clip_select_action_num_candidates > 1:
                    if cfg.clip_select_action_strategy == "highest_score":
                        valid_indices = np.where(combined_scores > -np.inf)[0]
                        if len(valid_indices) == 0:
                            print("  Warning: All candidate scores are invalid (-inf). Using first action.")
                            current_vla_clip_score = -np.inf
                            current_history_for_scoring = padded_histories[0] if padded_histories[0] is not None else None
                        else:
                            scores_valid = combined_scores[valid_indices]
                            best_valid_idx_in_valid_list = np.argmax(scores_valid)
                            best_candidate_idx = valid_indices[best_valid_idx_in_valid_list]
                            all_selected_instructions.append(batch_instructions[best_candidate_idx])
                            action_to_execute = predicted_actions[best_candidate_idx]
                            current_vla_clip_score = scores[best_candidate_idx]  # Original VLA-CLIP score
                            current_history_for_scoring = padded_histories[best_candidate_idx] if padded_histories[best_candidate_idx] is not None else None
                    elif cfg.clip_select_action_strategy == "softmax_sample":
                        # Implement softmax sampling with consistency-based temperature
                        valid_indices = np.where(combined_scores > -np.inf)[0]
                        if len(valid_indices) == 0:
                            print("  Warning: All candidate scores are invalid (-inf). Using first action.")
                            action_to_execute = predicted_actions[0]
                            current_vla_clip_score = -np.inf
                            current_history_for_scoring = padded_histories[0] if padded_histories[0] is not None else None
                        else:
                            scores_valid = combined_scores[valid_indices]
                            # Apply temperature scaling for softmax
                            scaled_scores = scores_valid / current_softmax_temperature
                            probs = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores))
                            # log_file.write(f"  [t={t}] Softmax sampling with temperature={current_softmax_temperature:.3f}\n")
                            
                            selected_idx = np.random.choice(len(valid_indices), p=probs)
                            best_candidate_idx = valid_indices[selected_idx]
                            all_selected_instructions.append(batch_instructions[best_candidate_idx])
                            action_to_execute = predicted_actions[best_candidate_idx]
                            current_vla_clip_score = scores[best_candidate_idx]  # Original VLA-CLIP score
                            current_history_for_scoring = padded_histories[best_candidate_idx] if padded_histories[best_candidate_idx] is not None else None
                else:
                    current_vla_clip_score = scores[0]
                    current_history_for_scoring = padded_histories[0] if padded_histories[0] is not None else None


                # --- Execute Action and Update State ---
                obs, reward, done, trunc, info = env.step(action_to_execute)
                
                # IMPORTANT: Append the action *actually executed* to the history
                executed_action_history.append(np.array(action_to_execute))

                # Append the score of the executed action for logging
                all_scores.append(current_vla_clip_score if not np.isnan(current_vla_clip_score) and current_vla_clip_score > -np.inf else np.nan)
                all_actions.append(action_to_execute)
                all_consistency_scores.append(consistency_score if not np.isnan(consistency_score) else np.nan)
                
                if done:
                    task_successes += 1
                    total_successes += 1
                    pbar.update(max_steps-t)
                    break
                t += 1
                pbar.update(1)
                # Update progress bar with scores
                postfix = {}
                if not np.isnan(current_vla_clip_score) and current_vla_clip_score > -np.inf:
                    postfix["score"] = f"{current_vla_clip_score:.3f}"
                else:
                    postfix["score"] = "N/A"
                
                if cfg.use_consistency_check and not np.isnan(consistency_score):
                    postfix["consistency"] = f"{consistency_score:.3f}"
                    postfix["temperature"] = f"{current_softmax_temperature:.3f}"
                elif cfg.use_consistency_check:
                    postfix["consistency"] = "N/A"
                    postfix["temperature"] = "N/A"
                
                pbar.set_postfix(postfix)

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
                clip_update_num=cfg.clip_select_action_num_candidates,
                consistency_indicator=cfg.use_consistency_check,
                all_consistency_scores=all_consistency_scores,
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
            avg_consistency = np.nanmean(all_consistency_scores) if all_consistency_scores else np.nan
            
            if cfg.use_consistency_check:
                print(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgStepScore={avg_score:.3f}, AvgConsistency={avg_consistency:.3f}")
                log_file.write(f"  Episode {total_episodes}: Success={done}, Steps={t-cfg.num_steps_wait}, AvgStepScore={avg_score:.3f}, AvgConsistency={avg_consistency:.3f}\n")
            else:
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