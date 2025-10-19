import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import os
import sys
import numpy as np
import random
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import TextAwareVisualExtraction, ModelConfig
from finetune_trajectory_bridge_ddp import VLA_SigLIP2_Bridge
import argparse
import json

ACTION_PADDING_VALUE = -5.0  # Define padding value globally


class VLA_SigLIP_Ensemble_Bridge_Inference:
    def __init__(self, model_configs, history_length, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize VLA-SigLIP ensemble inference model for Bridge dataset
        
        Args:
            model_configs: List of tuples (model_path, backbone, use_transformer)
            history_length: Action history length used during training
            device: Device to run inference on
        """
        self.device = device
        self.history_length = history_length
        self.models = []
        self.siglip_models = []
        self.preprocessors = []
        self.tokenizers = []
        
        # Deduplicate model configs based on model_path to avoid loading same model multiple times
        unique_configs = []
        seen_paths = set()
        for config in model_configs:
            if config[0] not in seen_paths:
                unique_configs.append(config)
                seen_paths.add(config[0])
        
        print(f"Loading {len(unique_configs)} unique model(s) for ensemble (from {len(model_configs)} specified)...")
        
        for i, (model_path, backbone, use_transformer) in enumerate(unique_configs):
            print(f"\n[Model {i+1}/{len(unique_configs)}] Loading: {os.path.basename(model_path)}")
            print(f"  Backbone: {backbone}")
            print(f"  Use Transformer: {use_transformer}")
            
            # Load the base SigLIP2 model and preprocessing
            siglip_model, preprocess = create_model_from_pretrained(backbone)
            siglip_model = siglip_model.to(device)
            siglip_model.eval()
            
            # Load the tokenizer
            tokenizer = get_tokenizer(backbone)
            
            # Initialize the VLA_SigLIP2_Bridge model
            # ModelConfig needs history_length and action_dim (7 for Bridge)
            model_config = ModelConfig(
                clip_model=siglip_model, 
                history_length=history_length, 
                action_dim=7
            )
            
            # Initialize model using VLA_SigLIP2_Bridge from training script
            # Note: VLA_SigLIP2_Bridge constructor already sets frozen encoder to bf16
            model = VLA_SigLIP2_Bridge(model_config, use_transformer=use_transformer).to(device)
            
            # Load trained weights (precision is preserved from checkpoint)
            print(f"  Loading model weights from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                # Handle both full checkpoint and state_dict only formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"  Loaded checkpoint from epoch {epoch}")
                    
                    # Display additional checkpoint information if available
                    if 'train_loss' in checkpoint and 'val_loss' in checkpoint:
                        print(f"    Training loss: {checkpoint['train_loss']:.4f}")
                        print(f"    Validation loss: {checkpoint['val_loss']:.4f}")
                    
                    if 'train_img2act_top1_acc' in checkpoint:
                        print(f"    Training img2act accuracy: {checkpoint['train_img2act_top1_acc']:.3f}")
                    if 'val_img2act_top1_acc' in checkpoint:
                        print(f"    Validation img2act accuracy: {checkpoint['val_img2act_top1_acc']:.3f}")
                    
                    if 'best_val_loss' in checkpoint:
                        print(f"    Best validation loss: {checkpoint['best_val_loss']:.4f}")
                    
                    if 'global_step' in checkpoint:
                        print(f"    Global step: {checkpoint['global_step']}")
                else:
                    # Assume it's just a state_dict (legacy format)
                    print("  Loading legacy checkpoint format (model weights only)")
                    model.load_state_dict(checkpoint)
                print("  Model loaded successfully.")
            except Exception as e:
                print(f"  Error loading model: {e}")
                raise
            
            model.eval()  # Set to evaluation mode
            # Precision: Frozen encoder in bf16, trainable parts in fp32 (as preserved from checkpoint)
            
            self.models.append(model)
            self.siglip_models.append(siglip_model)
            self.preprocessors.append(preprocess)
            self.tokenizers.append(tokenizer)
        
        print(f"\nSuccessfully loaded {len(self.models)} models for ensemble!")
    
    def get_embeddings(self, model_idx, image, instruction, action_history):
        """
        Extract normalized embeddings from a specific model
        
        Args:
            model_idx: Index of the model to use
            image: PIL Image or numpy array (single agent view image)
            instruction: String instruction
            action_history: Numpy array action history (H, D)
            
        Returns:
            image_text_embedding: Tensor of shape (512,) - normalized image+text embedding
            action_embedding: Tensor of shape (512,) - normalized action embedding
        """
        model = self.models[model_idx]
        preprocess = self.preprocessors[model_idx]
        tokenizer = self.tokenizers[model_idx]
        siglip_model = self.siglip_models[model_idx]
        
        # Preprocess the image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        img_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        # Tokenize instruction
        if isinstance(instruction, str):
            text_tokens = tokenizer([instruction], context_length=siglip_model.context_length).to(self.device)
        else:
            text_tokens = instruction.to(self.device)
            if text_tokens.ndim == 1:
                text_tokens = text_tokens.unsqueeze(0)
        
        # Prepare action history
        history_tensor = torch.tensor(action_history, dtype=torch.float32).unsqueeze(0).to(self.device)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        
        with torch.no_grad():
            # Extract features using the model's forward method
            # We need to extract intermediate embeddings before logit calculation
            
            # Extract image/text features
            patch_features, text_features = model.extract_features(img_tensor, text_tokens)
            
            # Text-aware visual features
            text_aware_features = model.text_aware_visual_extraction(patch_features, text_features)
            vision_token = model.vision_poolings(text_aware_features)
            
            # Text pooling
            text_token = model.text_pooling(text_features)
            combined_features = torch.cat([text_token, vision_token], dim=-1)
            combined_features = model.input_projection(combined_features)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            # Encode Action History
            action_histories = history_tensor.float()
            
            if model.use_transformer:
                # Transformer Path
                padding_mask = (action_histories[:, :, 0] == model.action_padding_value)
                encoded_steps = model.single_step_action_encoder(action_histories)
                encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
                transformer_output_permuted = model.trajectory_encoder(encoded_steps_permuted, src_key_padding_mask=padding_mask)
                transformer_output = transformer_output_permuted.permute(1, 0, 2)
                mask_expanded = (~padding_mask).unsqueeze(-1).float()
                summed_features = (transformer_output * mask_expanded).sum(dim=1)
                num_non_padded = mask_expanded.sum(dim=1)
                num_non_padded = torch.clamp(num_non_padded, min=1e-9)
                projected_trajectory = summed_features / num_non_padded
            else:
                # MLP Path
                batch_size = action_histories.shape[0]
                flat_actions = action_histories.reshape(batch_size, -1)
                projected_trajectory = model.complex_action_encoder(flat_actions)
            
            # Normalize action embedding
            projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
            
            # Return 512-d embeddings
            return combined_features.squeeze(0), projected_trajectory.squeeze(0)
    
    def fuse_embeddings(self, image, instruction, action_histories):
        """
        Fuse embeddings from all models according to the fusion procedure:
        1. L2-normalize each 512-d embedding from each model
        2. Average across all models
        3. L2-normalize again
        
        Args:
            image: PIL Image or numpy array
            instruction: String instruction
            action_histories: List of action history arrays
            
        Returns:
            fused_image_text_embeddings: Tensor of shape (num_histories, 512)
            fused_action_embeddings: Tensor of shape (num_histories, 512)
        """
        num_histories = len(action_histories)
        num_models = len(self.models)
        
        # Initialize storage for embeddings from all models
        all_image_text_embeds = []
        all_action_embeds = []
        
        # Extract embeddings from each model for all action histories
        for model_idx in range(num_models):
            model_image_text_embeds = []
            model_action_embeds = []
            
            for action_hist in action_histories:
                img_text_emb, action_emb = self.get_embeddings(model_idx, image, instruction, action_hist)
                # Embeddings are already L2-normalized (step 1)
                model_image_text_embeds.append(img_text_emb)
                model_action_embeds.append(action_emb)
            
            # Stack embeddings: (num_histories, 512)
            all_image_text_embeds.append(torch.stack(model_image_text_embeds))
            all_action_embeds.append(torch.stack(model_action_embeds))
        
        # Stack across models: (num_models, num_histories, 512)
        all_image_text_embeds = torch.stack(all_image_text_embeds)
        all_action_embeds = torch.stack(all_action_embeds)
        
        # Step 2: Average across all models
        fused_image_text = all_image_text_embeds.mean(dim=0)  # (num_histories, 512)
        fused_action = all_action_embeds.mean(dim=0)  # (num_histories, 512)
        
        # Step 3: L2-normalize again
        fused_image_text = fused_image_text / fused_image_text.norm(dim=-1, keepdim=True)
        fused_action = fused_action / fused_action.norm(dim=-1, keepdim=True)
        
        return fused_image_text, fused_action
    
    def predict(self, image, instruction, possible_action_histories):
        """
        Predict the most likely action history using ensemble fusion
        
        Args:
            image: PIL Image or numpy array (single agent view image)
            instruction: String instruction
            possible_action_histories: List of numpy action history arrays (Shape [H, D])
            
        Returns:
            predicted_history: The most likely action history (numpy array)
            history_scores: Dictionary mapping history index (str) to score (float)
        """
        # Get fused embeddings for all action histories
        fused_image_text, fused_action = self.fuse_embeddings(image, instruction, possible_action_histories)
        
        # Compute cosine similarity scores
        # Since embeddings are L2-normalized, dot product = cosine similarity
        scores = torch.matmul(fused_image_text, fused_action.T).diagonal()
        scores = scores.cpu().numpy()
        
        predicted_idx = scores.argmax()
        predicted_history = possible_action_histories[predicted_idx]
        
        history_scores = {str(i): float(scores[i]) for i in range(len(scores))}
        return predicted_history, history_scores


def sample_and_test_bridge_ensemble(bridge_dataset_dict, model_configs, history_length,
                                    num_samples=10, action_pool_size=20, images_folder=None):
    """
    Sample and test on Bridge dataset with ensemble of SigLIP models
    
    Args:
        bridge_dataset_dict: Dictionary loaded from bridge dataset JSON
        model_configs: List of tuples (model_path, backbone, use_transformer)
        history_length: Action history length
        num_samples: Number of samples to test
        action_pool_size: Size of action pool for each test
        images_folder: Path to images folder
    """
    inference_model = VLA_SigLIP_Ensemble_Bridge_Inference(
        model_configs,
        history_length=history_length
    )
    
    # Extract samples from bridge dataset format
    action_histories = bridge_dataset_dict['action_histories']
    instructions = bridge_dataset_dict['instructions']
    samples = bridge_dataset_dict['samples']
    
    print("Processing bridge dataset samples...")
    all_samples = []
    all_histories = []
    
    for sample in tqdm(samples, desc="Processing samples"):
        action_history_id = sample.get('action_history_id')
        instruction_id = sample.get('instruction_id')
        agent_view_image_file = sample.get('agent_view_image_file')
        
        if not all([action_history_id, instruction_id, agent_view_image_file]):
            continue
            
        # Get action history and instruction
        action_hist = np.array(action_histories[action_history_id])
        instruction = instructions[instruction_id]
        
        # Load image
        if images_folder:
            image_path = os.path.join(images_folder, agent_view_image_file)
        else:
            image_path = os.path.join(os.path.dirname(model_configs[0][0]), '../../10episodes_imgs', agent_view_image_file)
        
        if not os.path.exists(image_path):
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        all_samples.append((image, instruction, action_hist))
        all_histories.append(action_hist)
    
    if not all_samples:
        print("Error: No valid samples found in the dataset.")
        return []
    
    if not all_histories:
        print("Error: No valid action histories found for pooling.")
        return []
    
    print(f"Total samples for testing: {len(all_samples)}")
    print(f"Total histories for pooling: {len(all_histories)}")
    
    random.seed(42)
    sampled_indices = random.sample(range(len(all_samples)), min(num_samples, len(all_samples)))
    results = []
    
    for idx in tqdm(sampled_indices, desc="Testing samples with ensemble"):
        image, gt_instruction, gt_action_hist = all_samples[idx]
        
        # Create action history pool
        action_history_pool = [gt_action_hist]
        gt_pos_hist_added = True
        num_needed = action_pool_size - len(action_history_pool)
        
        if num_needed > 0:
            candidate_pool = [h for h in all_histories if not np.array_equal(h, gt_action_hist)]
            if len(candidate_pool) > 0:
                num_to_sample = min(num_needed, len(candidate_pool))
                sampled_histories = random.sample(candidate_pool, num_to_sample)
                action_history_pool.extend(sampled_histories)
            else:
                print(f"Warning: Not enough unique histories in dataset to fill pool for sample {idx}. Pool size: {len(action_history_pool)}")
        
        random.shuffle(action_history_pool)
        
        # Find ground truth index in pool
        ground_truth_idx_in_pool = None
        for i, hist in enumerate(action_history_pool):
            if np.array_equal(hist, gt_action_hist):
                ground_truth_idx_in_pool = i
                break
        
        if ground_truth_idx_in_pool is None and gt_pos_hist_added:
            print(f"Warning: Ground truth action history lost during pooling/shuffling for sample {idx}? This shouldn't happen.")
        
        # Make prediction with ensemble
        predicted_history, scores = inference_model.predict(
            image, gt_instruction, action_history_pool
        )
        
        is_correct = np.array_equal(predicted_history, gt_action_hist)
        
        results.append({
            'image': image,
            'instruction': gt_instruction,
            'ground_truth_action': gt_action_hist,
            'ground_truth_idx': ground_truth_idx_in_pool,
            'prediction': predicted_history,
            'action_pool_size': len(action_history_pool),
            'scores': scores,
            'correct': is_correct,
        })
    
    return results


def display_results(results):
    """Display the results of the sample_and_test function"""
    correct = 0
    ranks = []
    l2_distances = []

    print("\n--- Ensemble Evaluation Results ---")
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Instruction: {result['instruction']}")
        print(f"  Action pool size: {result['action_pool_size']}")
        print(f"  Ground truth action index in pool: {result['ground_truth_idx']}")

        print(f"  Correct (Predicted == GT Action): {result['correct']}")

        # Calculate L2 distance between prediction and ground truth action history
        if isinstance(result['prediction'], np.ndarray) and isinstance(result['ground_truth_action'], np.ndarray):
            pred_flat = result['prediction'].flatten()
            gt_flat = result['ground_truth_action'].flatten()
            l2_dist = np.linalg.norm(pred_flat - gt_flat)
            print(f"  L2 distance (Pred vs GT Action): {l2_dist:.4f}")
            l2_distances.append(l2_dist)

        # Display top predictions from the pool
        scores = result['scores']
        try:
            # Convert string keys to int for sorting if necessary
            scores_int_keys = {int(k): v for k, v in scores.items()}
            sorted_scores = sorted(scores_int_keys.items(), key=lambda item: item[1], reverse=True)
            print("  Top predictions (pool_index: score):")
            for pool_idx, score in sorted_scores[:5]:
                print(f"    {pool_idx}: {score:.4f}")

            # Calculate rank of ground truth action history
            if result['ground_truth_idx'] is not None:
                gt_score = scores_int_keys.get(result['ground_truth_idx'], -float('inf'))
                rank = sum(1 for score in scores_int_keys.values() if score > gt_score) + 1
                ranks.append(rank)
                print(f"  Rank of Ground Truth Action: {rank}")

        except ValueError:
            print("  Error processing scores for ranking (invalid format?).")

        if result['correct']:
            correct += 1

    # Calculate overall accuracy and mean rank
    accuracy = correct / len(results) if results else 0
    mean_rank = np.mean(ranks) if ranks else float('nan')
    mean_l2 = np.mean(l2_distances) if l2_distances else float('nan')

    print("-" * 25)
    print(f"Overall accuracy: {accuracy:.3f} ({correct}/{len(results)})")
    print(f"Mean rank of ground truth action history: {mean_rank:.3f}")
    print(f"Mean L2 distance (Prediction vs GT Action): {mean_l2:.4f}")
    print("-" * 25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VLA-SigLIP Ensemble Bridge Dataset Inference')
    
    # Model configurations - flexible approach supporting any number of models
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints (space-separated)')
    parser.add_argument('--backbones', type=str, nargs='+', required=True,
                       help='Backbone for each model (space-separated, must match number of model_paths)')
    parser.add_argument('--use_transformer', action='store_true',
                       help='All models use transformer encoder (if not set, all use MLP)')
    
    parser.add_argument('--history_length', type=int, default=10,
                       help='Action history length used during training (must match dataset)')

    # Dataset
    parser.add_argument('--bridge_dataset', type=str, required=True,
                       help='Path to the bridge dataset .json file')
    parser.add_argument('--images_folder', type=str, required=True,
                       help='Path to folder containing agent view images')

    # Evaluation Params
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test for evaluation')
    parser.add_argument('--action_pool_size', type=int, default=20,
                       help='Size of the action history pool (including GT) for each test sample')

    args = parser.parse_args()

    # Validate that number of backbones matches number of model paths
    if len(args.model_paths) != len(args.backbones):
        print(f"Error: Number of model_paths ({len(args.model_paths)}) must match number of backbones ({len(args.backbones)})")
        exit(1)

    # Create model configurations list
    model_configs = [
        (model_path, backbone, args.use_transformer)
        for model_path, backbone in zip(args.model_paths, args.backbones)
    ]
    
    print(f"Setting up ensemble with {len(model_configs)} model(s)")

    # Verify all model paths exist
    for model_path, _, _ in model_configs:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            exit(1)

    # Load bridge dataset
    print(f"Loading bridge dataset: {args.bridge_dataset}")
    if not os.path.exists(args.bridge_dataset):
        print(f"Error: Dataset file not found at {args.bridge_dataset}")
        exit(1)
    
    try:
        with open(args.bridge_dataset, 'r') as f:
            dataset_dict = json.load(f)
        print(f"Loaded bridge dataset with {len(dataset_dict.get('samples', []))} samples.")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)

    # Verify images folder
    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder not found at {args.images_folder}")
        exit(1)

    # Run evaluation
    print(f"\nStarting ensemble evaluation on Bridge dataset with SigLIP...")
    print(f"History length: {args.history_length}")
    print(f"Number of models in ensemble: {len(model_configs)}")
    
    results = sample_and_test_bridge_ensemble(
        bridge_dataset_dict=dataset_dict,
        model_configs=model_configs,
        history_length=args.history_length,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size,
        images_folder=args.images_folder
    )

    # Display results
    display_results(results)
