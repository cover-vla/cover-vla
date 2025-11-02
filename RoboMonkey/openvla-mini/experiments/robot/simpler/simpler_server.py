"""
Server API for SIMPLER evaluation pipeline.

Usage:
    conda activate <env> && \
    cd ~/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler && \
    python simpler_server.py
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
    verifier_path = os.getenv('VERIFIER_PATH', '/root/vla-clip/bridge_verifier/ensemble_789_trainable_only.pt')
    
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

@app.route('/process_action', methods=['POST'])
def process_action():
    """Main endpoint for processing action requests."""
    global pi0_policy, ensemble_model, preprocess_adapter
    
    # Initialize models if not already loaded
    if pi0_policy is None:
        initialize_models()
    
    # Get data from request
    data = request.json
    instruction = data.get('instruction')
    image_path = data.get('image_path')  # Path to image file instead of base64
    observation_state = data.get('observation_state')  # [7] array
    action_history = data.get('action_history', [])  # List of [7] arrays
    rephrased_list = data.get('rephrased_list', None)  # Optional list of rephrased instructions
    lang_rephrase_num = int(data.get('lang_rephrase_num', 1))
    policy_batch_inference_size = int(data.get('policy_batch_inference_size', 2))
    use_verifier = data.get('use_verifier', True)
    action_queue = data.get('action_queue', None)  # Optional existing action queue
    timestep = int(data.get('timestep', 0))
    n_action_steps = int(data.get('n_action_steps', 4))
    action_ensemble_temp = float(data.get('action_ensemble_temp', -0.8))
    
    if not instruction or not image_path or observation_state is None:
        return jsonify({'error': 'Missing required fields: instruction, image_path, or observation_state'}), 400
    
    # Convert lists back to numpy arrays (JSON deserializes arrays as lists)
    if isinstance(observation_state, dict) and 'agent' in observation_state:
        if 'eef_pos' in observation_state['agent']:
            observation_state['agent']['eef_pos'] = np.array(observation_state['agent']['eef_pos'])
    
    # Load image from disk path
    if not os.path.exists(image_path):
        return jsonify({'error': f'Image file not found: {image_path}'}), 400
    
    image = Image.open(image_path)
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
            output_action_queue = pi0_policy.select_action(observation, noise_std=1.7)
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
            
            # Update action history
            processed_action_for_history = max_action_history[num_past].copy()
            
            response_data['action'] = execute_action.tolist()
            response_data['selected_instruction'] = max_instruction
            response_data['verifier_score'] = float(max_score)
            response_data['action_queue'] = [
                aq.cpu().numpy().tolist() if isinstance(aq, torch.Tensor) else aq.tolist() 
                for aq in action_queue
            ]
            response_data['action_history_update'] = processed_action_for_history.tolist()
        else:
            # No verifier - use first action from queue
            single_action = action_queue.popleft().cpu().numpy()
            action_for_env = single_action[0:1]
            execute_action = convert_maniskill_with_bridge_adapter(
                action_for_env, verifier_action=False, action_ensemble_temp=action_ensemble_temp
            )
            
            response_data['action'] = execute_action.tolist()
            response_data['selected_instruction'] = instruction  # Keep original instruction
            response_data['verifier_score'] = None
            response_data['action_queue'] = [
                aq.cpu().numpy().tolist() if isinstance(aq, torch.Tensor) else aq.tolist() 
                for aq in action_queue
            ]
            response_data['action_history_update'] = None  # No history update when verifier not used
    else:
        # Use existing action queue
        if action_queue:
            # Convert action_queue back to deque
            action_queue_deque = deque()
            for aq_item in action_queue:
                if isinstance(aq_item, list):
                    aq_tensor = torch.tensor(aq_item, device=policy_device)
                else:
                    aq_tensor = aq_item
                action_queue_deque.append(aq_tensor)
            
            single_action = action_queue_deque.popleft().cpu().numpy()
            action_for_env = single_action[0:1]
            execute_action = convert_maniskill_with_bridge_adapter(
                action_for_env, verifier_action=False, action_ensemble_temp=action_ensemble_temp
            )
            
            response_data['action'] = execute_action.tolist()
            response_data['selected_instruction'] = instruction  # Keep same instruction when not at boundary
            response_data['verifier_score'] = None  # No verifier score when not at boundary
            response_data['action_queue'] = [
                aq.cpu().numpy().tolist() if isinstance(aq, torch.Tensor) else aq.tolist() 
                for aq in action_queue_deque
            ]
            
            # Update action history if verifier is used (matching original - updates even when not at boundary)
            if use_verifier and ensemble_model is not None:
                processed_action_for_history = convert_maniskill_with_bridge_adapter(
                    single_action[0:1], verifier_action=True, action_ensemble_temp=action_ensemble_temp
                )
                response_data['action_history_update'] = processed_action_for_history.tolist()
            else:
                response_data['action_history_update'] = None
        else:
            return jsonify({'error': 'No action queue provided and not at n_action_steps boundary'}), 400
    
    return jsonify(response_data)

if __name__ == "__main__":
    # Initialize models on startup
    initialize_models()
    
    # Run the server
    port = int(os.getenv('PORT', 5001))
    logger.info(f"Starting SIMPLER server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)

