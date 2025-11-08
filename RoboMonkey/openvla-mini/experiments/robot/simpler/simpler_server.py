"""
Server API for SIMPLER evaluation pipeline.

Key Features:
    - Server-side action_queue storage: Action queues are stored server-side per session/episode,
      eliminating the need for clients to send action_queue back and forth. Clients only need to
      provide a consistent session_id (or episode_id) across requests.
    
    - Server-side rephrased_list loading: Rephrased instructions are automatically loaded from JSON
      files based on the instruction text, eliminating the need for clients to load and pass them.

Usage:
    conda activate <env> && \
    cd ~/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler && \
    python simpler_server.py
    
Environment Variables:
    - REPHRASED_JSON_PATH: Path to rephrased instructions JSON file (default: simpler_rephrased_final_eval_vlm.json)
    - PRETRAINED_CHECKPOINT: Model checkpoint path
    - USE_VERIFIER: Whether to use verifier (default: True)
    - N_ACTION_STEPS: Number of action steps (default: 4)
    - ACTION_ENSEMBLE_TEMP: Action ensemble temperature (default: -0.8)
    - VERIFIER_PATH: Path to verifier model
    - PORT: Server port (default: 5001)
    - HOST: Server host (default: 0.0.0.0)
"""

import os
import sys
import json
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from collections import deque

# Import torch after setting up paths
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and components
pi0_policy = None
ensemble_model = None
preprocess_adapter = None

# Session storage for action queues (keyed by batch_number)
action_queue_storage = {}
# Storage for action histories - single history that persists across batches (maintains last 6 actions)
# Assumes episodes are sequential and always start at timestep 0
# Reset when timestep == 0 (new episode starts)
current_action_history = []

# Load rephrased instructions cache
rephrased_instructions_cache = None
rephrased_json_path = None

# Image files are now saved by client and passed as paths
# No need for server-side upload folder

# Import required modules
sys.path.append('/root/vla-clip/lerobot_intact')
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

sys.path.append('/root/vla-clip/bridge_verifier/ensemble_eval')
from efficient_ensemble_merged import EfficientEnsembleMerged

from experiments.robot.simpler.simpler_utils_robomonkey import (
    convert_maniskill_with_bridge_adapter,
    create_bridge_adapter_wrapper,
    process_raw_image_to_jpg,
)

def load_rephrased_instructions():
    """Load rephrased instructions from JSON file."""
    global rephrased_instructions_cache, rephrased_json_path
    
    if rephrased_instructions_cache is not None:
        return rephrased_instructions_cache
    
    # Get path from environment or use default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rephrased_json_path = os.getenv('REPHRASED_JSON_PATH', 
                                     os.path.join(script_dir, 'simpler_rephrased_final_eval_vlm.json'))
    
    if not os.path.exists(rephrased_json_path):
        raise FileNotFoundError(f"Rephrased JSON file not found at {rephrased_json_path}")
    
    with open(rephrased_json_path, 'r') as f:
        all_rephrases = json.load(f)
    rephrased_instructions_cache = all_rephrases.get("instructions", {})
    logger.info(f"Loaded {len(rephrased_instructions_cache)} rephrased instruction sets from {rephrased_json_path}")
    return rephrased_instructions_cache

def get_rephrased_list_for_instruction(original_instruction: str):
    """Get rephrased list for a given original instruction using exact match.
    
    Args:
        original_instruction: The original instruction text to match against 'rephrases_original' field
    
    Returns:
        List of rephrased instructions
    
    Raises:
        ValueError: If no matching rephrased instructions are found
    """
    rephrased_instructions = load_rephrased_instructions()
    
    if not rephrased_instructions:
        raise ValueError("No rephrased instructions loaded from JSON file")
    
    # Use exact match against rephrases_original field (normalized)
    normalized_original = original_instruction.lower().strip()
    
    for key, entry in rephrased_instructions.items():
        entry_original = entry.get("rephrases_original", "").lower().strip()
        if entry_original == normalized_original:
            return entry.get("ert_rephrases", [])
    
    raise ValueError(f"No rephrased instructions found for original instruction: {original_instruction}")

def initialize_models():
    """Initialize PI0 policy and ensemble verifier models."""
    global pi0_policy, ensemble_model, preprocess_adapter
    
    if pi0_policy is not None:
        logger.info("Models already initialized")
        return
    
    # Load configuration from environment or use defaults
    pretrained_checkpoint = os.getenv('PRETRAINED_CHECKPOINT', 'juexzz/INTACT-pi0-finetune-rephrase-bridge')
    use_verifier = os.getenv('USE_VERIFIER', 'True').lower() == 'true'
    n_action_steps = int(os.getenv('N_ACTION_STEPS', '4'))
    action_ensemble_temp = float(os.getenv('ACTION_ENSEMBLE_TEMP', '-0.8'))
    verifier_path = os.getenv('VERIFIER_PATH', '/root/vla-clip/bridge_verifier/ensemble_182123_trainable_only.pt')
    
    logger.info(f"Loading PI0 policy from {pretrained_checkpoint}...")
    pi0_policy = PI0Policy.from_pretrained(pretrained_checkpoint)
    
    if torch.cuda.is_available():
        pi0_policy.to("cuda")
        pi0_policy.config.device = "cuda"
    
    pi0_policy.config.n_action_steps = n_action_steps
    logger.info(f"PI0Policy device: {pi0_policy.config.device}")
    
    if use_verifier:
        logger.info(f"Loading ensemble model from {verifier_path}...")
        ensemble_model = EfficientEnsembleMerged(verifier_path)
        logger.info("Ensemble model loaded successfully!")
    
    # Create adapter for preprocessing
    preprocess_adapter = create_bridge_adapter_wrapper(action_ensemble_temp)
    pi0_policy._preprocess_adapter = preprocess_adapter
    
    # Preload rephrased instructions
    load_rephrased_instructions()
    
    logger.info("Models initialized successfully!")

def process_inputs(batch_size, predefined_action_queue, verifier_action=False, action_history=[], n_action_steps=4):
    """Process action inputs for verifier."""
    processed_future_actions_batch = []
    for i in range(n_action_steps):
        single_action = predefined_action_queue[i].cpu().numpy()
        processed_execution_actions_for_step = []
        for batch_idx in range(batch_size):
            sample_1x7 = single_action[batch_idx:batch_idx+1]
            processed_execution_1x7 = convert_maniskill_with_bridge_adapter(
                sample_1x7, verifier_action=verifier_action, action_ensemble_temp=-0.8
            )
            processed_execution_1x7 = np.asarray(processed_execution_1x7)
            processed_execution_actions_for_step.append(processed_execution_1x7)
        
        processed_batch = np.vstack(processed_execution_actions_for_step)
        processed_batch = processed_batch.reshape(batch_size, 7)
        processed_future_actions_batch.append(processed_batch)
    
    num_past = min(len(action_history), 6)
    future_actions = np.stack(processed_future_actions_batch)
    future_actions_transposed = future_actions.transpose(1, 0, 2)
    
    if num_past > 0:
        past_actions = np.stack(action_history[-num_past:])
        past_actions = np.expand_dims(past_actions, axis=0).repeat(batch_size, axis=0)
        processed_full_trajectory = np.concatenate([past_actions, future_actions_transposed], axis=1)
    else:
        processed_full_trajectory = future_actions_transposed
    
    action_histories_list = [processed_full_trajectory[i] for i in range(batch_size)]
    return action_histories_list

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions and print traceback."""
    import traceback
    error_traceback = traceback.format_exc()
    logger.error(f"Unhandled exception: {str(e)}\n{error_traceback}")
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"ERROR: {str(e)}", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(error_traceback, file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)
    return jsonify({'error': str(e), 'traceback': error_traceback}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'models_loaded': pi0_policy is not None})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset/clear action queue and history."""
    global action_queue_storage, current_action_history
    data = request.json
    timestep = int(data.get('timestep', 0))
    n_action_steps = 4
    batch_number = timestep // n_action_steps
    
    cleared_queue = batch_number in action_queue_storage
    
    if batch_number in action_queue_storage:
        del action_queue_storage[batch_number]
    # Reset action history
    current_action_history = []
    
    return jsonify({
        'status': 'success', 
        'message': f'Reset batch {batch_number} and cleared action history'
    })

@app.route('/process_action', methods=['POST'])
def process_action():
    """Main endpoint for processing action requests."""
    global pi0_policy, ensemble_model, preprocess_adapter, action_queue_storage
    
    # Initialize models if not already loaded
    if pi0_policy is None:
        initialize_models()
    
    # Get data from request
    data = request.json
    instruction = data.get('instruction')
    original_instruction = data.get('original_instruction', instruction)  # Use original_instruction if provided, fallback to instruction
    image_base64 = data.get('image')  # Base64 encoded image
    observation_state = data.get('observation_state')  # [7] array
    # Retrieve action_history from server-side storage instead of client input
    # Load rephrased_list from server-side JSON using exact match on original_instruction
    rephrased_list = get_rephrased_list_for_instruction(original_instruction) if original_instruction else None
    lang_rephrase_num = 8
    policy_batch_inference_size = 5
    use_verifier = True
    timestep = int(data.get('timestep', 0))
    n_action_steps = 4
    action_ensemble_temp = -0.8
    
    # Use batch number as key - timestep tells us which batch we're in
    # If timestep % n_action_steps == 0, we generate new actions
    # Otherwise, we use the stored queue from the previous request
    batch_number = timestep // n_action_steps
    
    # Retrieve action_queue from server-side storage
    action_queue = action_queue_storage.get(batch_number, None)
    
    # Reset action_history when starting a new episode (timestep == 0)
    global current_action_history
    if timestep == 0:
        current_action_history = []
    action_history = current_action_history  # Use the persistent history
    
    if not instruction or not image_base64 or observation_state is None:
        return jsonify({'error': 'Missing required fields: instruction, image, or observation_state'}), 400
    
    # Convert lists back to numpy arrays (JSON deserializes arrays as lists)
    if isinstance(observation_state, dict) and 'agent' in observation_state:
        if 'eef_pos' in observation_state['agent']:
            observation_state['agent']['eef_pos'] = np.array(observation_state['agent']['eef_pos'])
    
    # Load image from base64
    import base64
    import io
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    raw_img = np.array(image)
    
    # Prepare observation for adapter (matching original format)
    obs_for_adapter = {
        'observation.images.top': raw_img,
        'observation.state': observation_state,
        'task': instruction
    }
    
    # Preprocess observation
    processed_obs = preprocess_adapter.preprocess(obs_for_adapter)
    
    # Move to policy device
    policy_device = torch.device(pi0_policy.config.device)
    processed_obs = {
        k: (v.to(device=policy_device) if isinstance(v, torch.Tensor) else v)
        for k, v in processed_obs.items()
    }
    
    # Get image feature key
    image_feature_keys = list(pi0_policy.config.image_features.keys())
    image_key = image_feature_keys[0]
    
    # Build instruction batch
    batch_size = policy_batch_inference_size * lang_rephrase_num
    
    if rephrased_list is not None and lang_rephrase_num > 1:
        unique_prompts = [instruction] + rephrased_list[:lang_rephrase_num - 1]
    else:
        unique_prompts = [instruction]
    
    task_list = []
    for p in unique_prompts:
        task_list.extend([p] * policy_batch_inference_size)
    
    assert len(task_list) == batch_size, "Batch size mismatch"
    
    # Create batch observation
    batch_image = processed_obs['observation.images.top'].repeat(batch_size, 1, 1, 1)
    batch_state = processed_obs['observation.state'].repeat(batch_size, 1)
    
    observation = {
        image_key: batch_image,
        "observation.state": batch_state,
        "task": task_list,
    }
    
    # Generate actions or use existing queue
    response_data = {
        'status': 'success',
        'action': None,
        'selected_instruction': instruction,
        'verifier_score': None,
        'action_queue': None,
        'action_history_update': None
    }
    
    # Generate new actions if needed (at n_action_steps boundary)
    if timestep % n_action_steps == 0:
        with torch.no_grad():
            output_action_queue = pi0_policy.select_action(observation, noise_std=1.0)
            action_queue = output_action_queue.copy()
            output_action_queue.clear()
        
        if use_verifier and ensemble_model is not None:
            # Use verifier to select best action
            assert len(action_queue) == n_action_steps
            
            # Convert action_history to numpy arrays (matching original format)
            np_action_history = [np.array(ah) for ah in action_history] if action_history else []
            
            # Calculate num_past BEFORE processing (matching original)
            num_past = min(len(np_action_history), 6)
            
            predefined_action_queue = list(action_queue)
            action_queue.popleft()
            
            action_histories_list = process_inputs(
                batch_size, predefined_action_queue, 
                verifier_action=True, 
                action_history=np_action_history.copy(), 
                n_action_steps=n_action_steps
            )
            
            images_list = [process_raw_image_to_jpg(raw_img)] * batch_size
            
            max_score, max_instruction, max_action_history, global_action_idx = ensemble_model.compute_max_similarity_scores_batch(
                images=images_list[0:policy_batch_inference_size],
                instructions=[instruction] * policy_batch_inference_size,
                all_action_histories=action_histories_list[0:policy_batch_inference_size],
                cfg_repeat_language_instructions=policy_batch_inference_size
            )
            
            if max_score < 0.1:
                max_score, max_instruction, max_action_history, global_action_idx = ensemble_model.compute_max_similarity_scores_batch(
                    images=images_list,
                    instructions=task_list,
                    all_action_histories=action_histories_list,
                    cfg_repeat_language_instructions=policy_batch_inference_size
                )
            
            # Get execution action
            execution_action_histories_list = process_inputs(
                batch_size, predefined_action_queue, 
                verifier_action=False, 
                action_history=np_action_history.copy(), 
                n_action_steps=n_action_steps
            )
            
            execute_action = execution_action_histories_list[global_action_idx][num_past].copy()
            
            # Vote on gripper state
            group_start = (global_action_idx // policy_batch_inference_size) * policy_batch_inference_size
            group_end = group_start + policy_batch_inference_size
            stacked_histories = np.stack(execution_action_histories_list[group_start:group_end])
            grippers = stacked_histories[:, num_past, -1]
            
            close_votes = int((grippers >= 0).sum())
            open_votes = int((grippers < 0).sum())
            
            if close_votes > open_votes:
                execute_action[-1] = 1.0
            elif open_votes > close_votes:
                execute_action[-1] = -1.0
            else:
                execute_action[-1] = 1.0 if execute_action[-1] >= 0 else -1.0
            
            execute_action[-1] = float(np.sign(execute_action[-1]))
            
            # Extract remaining actions from selected chunk
            selected_action_chunk = deque()
            for timestep_idx in range(1, n_action_steps):
                timestep_actions = predefined_action_queue[timestep_idx]
                selected_action = timestep_actions[global_action_idx:global_action_idx+1]
                selected_action_chunk.append(selected_action)
            
            action_queue = selected_action_chunk
            
            # Store action_queue in server-side storage
            action_queue_storage[batch_number] = action_queue
            
            # Update action history server-side
            processed_action_for_history = max_action_history[num_past].copy()
            # Append to stored history and keep only last 6 actions
            current_action_history.append(processed_action_for_history)
            current_action_history = current_action_history[-6:]  # Keep only last 6
            
            response_data['action'] = execute_action.tolist()
            response_data['selected_instruction'] = max_instruction
            response_data['verifier_score'] = float(max_score)
            response_data['action_queue'] = [
                aq.cpu().numpy().tolist() if isinstance(aq, torch.Tensor) else aq.tolist() 
                for aq in action_queue
            ]
            # No need to return action_history_update - client doesn't need it anymore
            response_data['action_history_update'] = None
        else:
            # No verifier - use first action from queue
            single_action = action_queue.popleft().cpu().numpy()
            action_for_env = single_action[0:1]
            execute_action = convert_maniskill_with_bridge_adapter(
                action_for_env, verifier_action=False, action_ensemble_temp=action_ensemble_temp
            )
            
            # Store remaining action_queue in server-side storage
            action_queue_storage[batch_number] = action_queue
            
            # Update action history server-side (even without verifier, for consistency)
            current_action_history.append(execute_action.copy())
            current_action_history = current_action_history[-6:]  # Keep only last 6
            
            response_data['action'] = execute_action.tolist()
            response_data['selected_instruction'] = instruction  # Keep original instruction
            response_data['verifier_score'] = None
            response_data['action_queue'] = [
                aq.cpu().numpy().tolist() if isinstance(aq, torch.Tensor) else aq.tolist() 
                for aq in action_queue
            ]
            response_data['action_history_update'] = None  # No need to return - tracked server-side
    else:
        # Use existing action queue from server-side storage
        if action_queue is None:
            return jsonify({'error': f'No action queue available for batch {batch_number} (timestep {timestep}). This should not happen if timestep is sequential.'}), 400
        
        # action_queue from storage should be a deque of tensors
        if not isinstance(action_queue, deque):
            # If somehow stored as list, convert back to deque
            action_queue_deque = deque()
            for aq_item in action_queue:
                if isinstance(aq_item, list):
                    aq_tensor = torch.tensor(aq_item, device=policy_device)
                else:
                    aq_tensor = aq_item
                action_queue_deque.append(aq_tensor)
            action_queue = action_queue_deque
        
        if len(action_queue) == 0:
            return jsonify({'error': 'Action queue is empty'}), 400
        
        single_action = action_queue.popleft()
        # Ensure it's a tensor
        if isinstance(single_action, list):
            single_action = torch.tensor(single_action, device=policy_device)
        single_action_np = single_action.cpu().numpy()
        action_for_env = single_action_np[0:1]
        execute_action = convert_maniskill_with_bridge_adapter(
            action_for_env, verifier_action=False, action_ensemble_temp=action_ensemble_temp
        )
        
        # Update stored action_queue
        action_queue_storage[batch_number] = action_queue
        
        response_data['action'] = execute_action.tolist()
        response_data['selected_instruction'] = instruction  # Keep same instruction when not at boundary
        response_data['verifier_score'] = None  # No verifier score when not at boundary
        response_data['action_queue'] = [
            aq.cpu().numpy().tolist() if isinstance(aq, torch.Tensor) else aq.tolist() 
            for aq in action_queue
        ]
        
        # Update action history server-side
        if use_verifier and ensemble_model is not None:
            processed_action_for_history = convert_maniskill_with_bridge_adapter(
                single_action_np[0:1], verifier_action=True, action_ensemble_temp=action_ensemble_temp
            )
            current_action_history.append(processed_action_for_history)
        else:
            current_action_history.append(execute_action.copy())
        current_action_history = current_action_history[-6:]  # Keep only last 6
        
        response_data['action_history_update'] = None  # No need to return - tracked server-side
    
    return jsonify(response_data)

if __name__ == "__main__":
    # Initialize models on startup
    initialize_models()
    
    # Run the server
    port = int(os.getenv('PORT', 5001))
    host = os.getenv('HOST', '0.0.0.0')  # Default to 0.0.0.0 to accept connections from any network interface
    logger.info(f"Starting SIMPLER server on {host}:{port}...")
    logger.info(f"Server accessible at http://<server_ip>:{port}")
    app.run(host=host, port=port, debug=False)

