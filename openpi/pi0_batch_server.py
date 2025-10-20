#!/usr/bin/env python3
"""
HTTP API server for Pi0Policy batch inference.

This server loads the Pi0Policy model and provides a batch inference endpoint
that processes multiple instructions with a single image and returns raw normalized actions.

Usage:
    python pi0_batch_server.py

Endpoint:
    POST http://localhost:3200/batch
    Body: {
        "instructions": ["instruction 1", "instruction 2", ...],
        "image_path": "/path/to/image.jpg",
        "temperature": 0.0  (optional, default: 0.0)
    }
    
    Returns: {
        "output_ids": [[...], [...], ...],  # Not used for Pi0, placeholder
        "actions": [[[...], [...], [...], [...]], ...]  # [num_instructions, 4_steps, 7_dims]
    }
"""

import sys
import os
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Add project paths
sys.path.append('/root/pi/vla-clip/lerobot_intact')
sys.path.append('/root/pi/vla-clip/INT-ACT')

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from src.experiments.env_adapters.simpler import BridgeSimplerAdapter


# Global variables for model and adapter
policy = None
adapter = None
device = None


class BatchRequest(BaseModel):
    instructions: List[str]
    image_path: str
    temperature: Optional[float] = 0.0


def create_bridge_adapter(action_ensemble_temp=-0.8):
    """Creates a BridgeSimplerAdapter for preprocessing and postprocessing."""
    class EnvConfig:
        def __init__(self):
            self.dataset_statistics_path = "/root/pi/vla-clip/INT-ACT/config/dataset/bridge_statistics.json"
            self.image_size = (224, 224)
            self.action_normalization_type = "bound"
            self.state_normalization_type = "bound"
    
    class ModelConfig:
        def __init__(self):
            self.chunk_size = 4  # Default action chunk size
            self.action_ensemble_temp = action_ensemble_temp
    
    class Config:
        def __init__(self):
            self.env = EnvConfig()
            self.use_bf16 = False
            self.seed = 42
            self.model_cfg = ModelConfig()
    
    config = Config()
    return BridgeSimplerAdapter(config)


def load_image_from_file(image_path):
    """Load image from file in RGB format as numpy array."""
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    return image_array


def create_dummy_robot_state():
    """Create a dummy robot state that mimics SimplerEnv observation."""
    position = np.array([0.3, 0.0, 0.1])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (wxyz)
    gripper = np.array([1.0])
    eef_pos = np.concatenate([position, quaternion, gripper])
    return {"agent": {"eef_pos": eef_pos}}


def initialize_model():
    """Initialize the Pi0Policy model and adapter."""
    global policy, adapter, device
    
    print("Initializing Pi0Policy server...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the pretrained Pi0 policy
    model_name = "juexzz/INTACT-pi0-finetune-bridge"
    print(f"Loading Pi0Policy from {model_name}...")
    
    policy = PI0Policy.from_pretrained(model_name)
    if torch.cuda.is_available():
        policy = policy.to(device)
        policy.config.device = str(device)
    policy.config.n_action_steps = 4  # 4 action steps
    policy.eval()
    
    print(f"Model loaded successfully!")
    print(f"  - Device: {policy.config.device}")
    print(f"  - Action steps: {policy.config.n_action_steps}")
    
    # Create the BridgeSimplerAdapter
    print("Creating BridgeSimplerAdapter...")
    adapter = create_bridge_adapter(action_ensemble_temp=-0.8)
    print("Adapter created successfully!")
    
    print("Server initialization complete!\n")


def generate_actions_for_batch(instructions: List[str], image_path: str, temperature: float = 0.0):
    """
    Generate actions for a batch of instructions with a single image.
    
    Args:
        instructions: List of instruction strings
        image_path: Path to the image file
        temperature: Temperature for sampling (not used in current implementation)
    
    Returns:
        Tuple of (output_ids, actions) where:
            - output_ids: Placeholder list of lists (not used for Pi0)
            - actions: numpy array of shape [num_instructions, 4, 7]
    """
    global policy, adapter, device
    
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if len(instructions) == 0:
        raise ValueError("At least one instruction is required")
    
    # Load the test image
    raw_img = load_image_from_file(image_path)
    
    # Create dummy robot state
    robot_obs = create_dummy_robot_state()
    
    # Process each instruction
    batch_size = len(instructions)
    all_actions = []
    
    for instruction in instructions:
        # Prepare observation for adapter
        obs_for_adapter = {
            'observation.images.top': raw_img,
            'observation.state': robot_obs,
            'task': instruction
        }
        
        # Preprocess with BridgeSimplerAdapter
        processed_obs = adapter.preprocess(obs_for_adapter)
        
        # Move to policy device
        policy_device = torch.device(policy.config.device)
        processed_obs = {
            k: (v.to(device=policy_device) if isinstance(v, torch.Tensor) else v)
            for k, v in processed_obs.items()
        }
        
        # Create batch observation (batch_size=1 for single instruction)
        image_feature_keys = list(policy.config.image_features.keys())
        image_key = image_feature_keys[0]
        
        # Adapter already returns tensors with batch dimension [1, ...]
        # No need to unsqueeze, just use them directly
        observation = {
            image_key: processed_obs['observation.images.top'],  # [1, 3, 224, 224]
            "observation.state": processed_obs['observation.state'],  # [1, 7]
            "task": processed_obs['task'],  # List of instructions
        }
        
        # Generate actions
        policy._action_queue.clear()
        
        with torch.no_grad():
            # Prepare inputs
            images, img_masks = policy.prepare_images(observation)
            state = policy.prepare_state(observation)
            lang_tokens, lang_masks = policy.prepare_language(observation)
            
            # Sample actions
            actions = policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=None
            )
            
            # Slice to get n_action_steps
            actions = actions[:, :policy.config.n_action_steps]
            
            # Slice to original action dimension
            original_action_dim = policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            
            # Unnormalize actions
            actions = policy.unnormalize_outputs({"action": actions})["action"]
            
            # Convert to numpy: [batch=1, n_steps=4, action_dim=7]
            actions_np = actions.cpu().numpy()
            
            # Extract the single batch item: [4, 7]
            action_steps = actions_np[0]
            all_actions.append(action_steps)
    
    # Stack all actions: [num_instructions, 4, 7]
    all_actions = np.stack(all_actions, axis=0)
    
    # Create placeholder output_ids (not used for Pi0, but needed for compatibility)
    output_ids = [[0] * 100 for _ in range(batch_size)]  # Placeholder
    
    return output_ids, all_actions


# FastAPI app
app = FastAPI(title="Pi0Policy Batch Inference Server")


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup."""
    initialize_model()


@app.post("/batch")
async def batch_inference(request: BatchRequest):
    """
    Batch inference endpoint for Pi0Policy.
    
    Accepts multiple instructions and a single image, returns raw normalized actions.
    """
    try:
        # Generate actions
        output_ids, actions = generate_actions_for_batch(
            instructions=request.instructions,
            image_path=request.image_path,
            temperature=request.temperature
        )
        
        # Convert numpy arrays to lists for JSON serialization
        return JSONResponse(content={
            "output_ids": output_ids,
            "actions": actions.tolist()
        })
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": policy is not None}


def main():
    print("="*80)
    print("Pi0Policy Batch Inference Server")
    print("="*80)
    print("Starting server on http://localhost:3200")
    print("Endpoint: POST /batch")
    print("="*80)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=3200, log_level="info")


if __name__ == "__main__":
    main()

