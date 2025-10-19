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

ACTION_PADDING_VALUE = -5.0


class EfficientVLA_SigLIP_Ensemble_Bridge:
    def __init__(self, model_paths, history_length, backbone='hf-hub:timm/ViT-L-16-SigLIP2-384',
                 use_transformer=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Efficient ensemble that shares the frozen encoder across all checkpoints.
        
        Args:
            model_paths: List of paths to model checkpoints
            history_length: Action history length
            backbone: SigLIP2 model backbone (shared across all models)
            use_transformer: Whether models use transformer encoder
            device: Device to run inference on
        """
        self.device = device
        self.history_length = history_length
        self.use_transformer = use_transformer
        self.num_models = len(model_paths)
        
        print(f"Loading shared SigLIP encoder: {backbone}")
        # Load the shared SigLIP2 model and preprocessing ONCE
        self.siglip_model, self.preprocess = create_model_from_pretrained(backbone)
        self.siglip_model = self.siglip_model.to(device)
        self.siglip_model.eval()
        
        # Freeze the encoder (as done in training)
        for param in self.siglip_model.parameters():
            param.requires_grad = False
        
        # Convert frozen encoder to bf16 for efficiency
        self.siglip_model = self.siglip_model.to(torch.bfloat16)
        
        # Load the tokenizer
        self.tokenizer = get_tokenizer(backbone)
        
        # Load trainable components from each checkpoint
        self.trainable_models = []
        self.full_model_for_features = None  # Will store first model for extract_features
        print(f"\nLoading trainable components from {len(model_paths)} checkpoints...")
        
        for i, model_path in enumerate(model_paths):
            print(f"\n[Model {i+1}/{len(model_paths)}] Loading: {os.path.basename(model_path)}")
            
            # Create model config
            model_config = ModelConfig(
                clip_model=self.siglip_model,
                history_length=history_length,
                action_dim=7
            )
            
            # Initialize full model
            full_model = VLA_SigLIP2_Bridge(model_config, use_transformer=use_transformer).to(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                full_model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"  Loaded checkpoint from epoch {epoch}")
                
                if 'val_img2act_top1_acc' in checkpoint:
                    print(f"  Validation accuracy: {checkpoint['val_img2act_top1_acc']:.3f}")
            else:
                full_model.load_state_dict(checkpoint)
            
            full_model.eval()
            
            # Save first full model for extract_features method
            if i == 0:
                self.full_model_for_features = full_model
            
            # Extract only the trainable components we need
            trainable_components = {
                'text_aware_visual_extraction': full_model.text_aware_visual_extraction,
                'vision_poolings': full_model.vision_poolings,
                'text_pooling': full_model.text_pooling,
                'input_projection': full_model.input_projection,
                'single_step_action_encoder': full_model.single_step_action_encoder if use_transformer else None,
                'trajectory_encoder': full_model.trajectory_encoder if use_transformer else None,
                'complex_action_encoder': full_model.complex_action_encoder if not use_transformer else None,
                'action_padding_value': full_model.action_padding_value,
            }
            
            self.trainable_models.append(trainable_components)
        
        print(f"\n✅ Successfully loaded shared encoder + {len(self.trainable_models)} trainable component sets!")
    
    def extract_shared_features(self, img_tensor, text_tokens):
        """
        Extract features using the full model's extract_features method.
        This properly handles dtype conversion and uses the shared frozen encoder.
        """
        with torch.no_grad():
            # Use the first full model's extract_features method
            # This handles all the dtype conversion and activation hooks properly
            patch_features, text_features = self.full_model_for_features.extract_features(img_tensor, text_tokens)
            return patch_features, text_features
    
    def get_embeddings_from_model(self, model_idx, patch_features, text_features, action_history):
        """
        Get embeddings using specific trainable components
        
        Args:
            model_idx: Index of the trainable model to use
            patch_features: Shared patch features from encoder
            text_features: Shared text features from encoder
            action_history: Action history tensor
            
        Returns:
            image_text_embedding: Normalized (512,) embedding
            action_embedding: Normalized (512,) embedding
        """
        components = self.trainable_models[model_idx]
        
        with torch.no_grad():
            # Process image/text with this model's trainable components
            # Features from extract_features are already in the correct dtype
            text_aware_features = components['text_aware_visual_extraction'](patch_features, text_features)
            vision_token = components['vision_poolings'](text_aware_features)
            text_token = components['text_pooling'](text_features)
            
            combined_features = torch.cat([text_token, vision_token], dim=-1)
            combined_features = components['input_projection'](combined_features)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            # Process action history with this model's trainable components
            action_histories = action_history.float()
            
            if self.use_transformer:
                # Transformer path
                padding_mask = (action_histories[:, :, 0] == components['action_padding_value'])
                encoded_steps = components['single_step_action_encoder'](action_histories)
                encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
                transformer_output_permuted = components['trajectory_encoder'](
                    encoded_steps_permuted, src_key_padding_mask=padding_mask
                )
                transformer_output = transformer_output_permuted.permute(1, 0, 2)
                mask_expanded = (~padding_mask).unsqueeze(-1).float()
                summed_features = (transformer_output * mask_expanded).sum(dim=1)
                num_non_padded = mask_expanded.sum(dim=1)
                num_non_padded = torch.clamp(num_non_padded, min=1e-9)
                projected_trajectory = summed_features / num_non_padded
            else:
                # MLP path
                batch_size = action_histories.shape[0]
                flat_actions = action_histories.reshape(batch_size, -1)
                projected_trajectory = components['complex_action_encoder'](flat_actions)
            
            # Normalize action embedding
            projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
            
            return combined_features.squeeze(0), projected_trajectory.squeeze(0)
    
    def get_embeddings_from_model_batch(self, model_idx, patch_features, text_features, action_histories_batch):
        """
        Get embeddings using specific trainable components (batched version)
        
        Args:
            model_idx: Index of the trainable model to use
            patch_features: Shared patch features (batch_size=1, num_patches, dim)
            text_features: Shared text features (batch_size=1, num_tokens, dim)
            action_histories_batch: Batch of action histories (batch_size, history_len, action_dim)
            
        Returns:
            combined_features: (batch_size, 512) - normalized image+text embeddings
            projected_trajectory: (batch_size, 512) - normalized action embeddings
        """
        components = self.trainable_models[model_idx]
        batch_size = action_histories_batch.shape[0]
        
        with torch.no_grad():
            # Repeat features for batch
            patch_features_batch = patch_features.repeat(batch_size, 1, 1)
            text_features_batch = text_features.repeat(batch_size, 1, 1)
            
            # Process image/text with this model's trainable components
            text_aware_features = components['text_aware_visual_extraction'](patch_features_batch, text_features_batch)
            vision_token = components['vision_poolings'](text_aware_features)
            text_token = components['text_pooling'](text_features_batch)
            
            combined_features = torch.cat([text_token, vision_token], dim=-1)
            combined_features = components['input_projection'](combined_features)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            # Encode Action Histories (batch)
            action_histories = action_histories_batch.float()
            
            if self.use_transformer:
                # Transformer Path
                padding_mask = (action_histories[:, :, 0] == components['action_padding_value'])
                encoded_steps = components['single_step_action_encoder'](action_histories)
                encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
                transformer_output_permuted = components['trajectory_encoder'](encoded_steps_permuted, src_key_padding_mask=padding_mask)
                transformer_output = transformer_output_permuted.permute(1, 0, 2)
                mask_expanded = (~padding_mask).unsqueeze(-1).float()
                summed_features = (transformer_output * mask_expanded).sum(dim=1)
                num_non_padded = mask_expanded.sum(dim=1)
                num_non_padded = torch.clamp(num_non_padded, min=1e-9)
                projected_trajectory = summed_features / num_non_padded
            else:
                # MLP Path
                flat_actions = action_histories.reshape(batch_size, -1)
                projected_trajectory = components['complex_action_encoder'](flat_actions)
            
            # Normalize action embedding
            projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
            
            return combined_features, projected_trajectory
    
    def fuse_embeddings(self, image, instruction, action_histories):
        """
        Fuse embeddings from all models:
        1. Extract features once using shared encoder (via extract_features method)
        2. Process through each model's trainable components
        3. L2-normalize each embedding
        4. Average across models
        5. L2-normalize again
        
        Args:
            image: PIL Image or numpy array
            instruction: String instruction
            action_histories: List of action history arrays
            
        Returns:
            fused_image_text_embeddings: Tensor of shape (num_histories, 512)
            fused_action_embeddings: Tensor of shape (num_histories, 512)
        """
        # Preprocess image and text ONCE
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        if isinstance(instruction, str):
            text_tokens = self.tokenizer([instruction], context_length=self.siglip_model.context_length).to(self.device)
        else:
            text_tokens = instruction.to(self.device)
            if text_tokens.ndim == 1:
                text_tokens = text_tokens.unsqueeze(0)
        
        # Extract shared features ONCE from frozen encoder
        patch_features, text_features = self.extract_shared_features(img_tensor, text_tokens)
        
        # Convert action histories to batch tensor
        action_histories_array = np.array(action_histories)
        action_histories_batch = torch.tensor(action_histories_array, dtype=torch.float32).to(self.device)
        
        # Initialize storage for embeddings from all models
        all_image_text_embeds = []
        all_action_embeds = []
        
        # Process through each model's trainable components in batch
        for model_idx in range(self.num_models):
            # Process all action histories in one forward pass
            img_text_embeds, action_embeds = self.get_embeddings_from_model_batch(
                model_idx, patch_features, text_features, action_histories_batch
            )
            # Embeddings are already L2-normalized
            all_image_text_embeds.append(img_text_embeds)
            all_action_embeds.append(action_embeds)
        
        # Stack across models: (num_models, num_histories, 512)
        all_image_text_embeds = torch.stack(all_image_text_embeds)
        all_action_embeds = torch.stack(all_action_embeds)
        
        # Average across all models
        fused_image_text = all_image_text_embeds.mean(dim=0)  # (num_histories, 512)
        fused_action = all_action_embeds.mean(dim=0)  # (num_histories, 512)
        
        # L2-normalize again
        fused_image_text = fused_image_text / fused_image_text.norm(dim=-1, keepdim=True)
        fused_action = fused_action / fused_action.norm(dim=-1, keepdim=True)
        
        return fused_image_text, fused_action
    
    def predict(self, image, instruction, possible_action_histories):
        """
        Predict the most likely action history using efficient ensemble fusion
        
        Args:
            image: PIL Image or numpy array
            instruction: String instruction
            possible_action_histories: List of numpy action history arrays
            
        Returns:
            predicted_history: The most likely action history
            history_scores: Dictionary mapping history index to score
        """
        # Get fused embeddings
        fused_image_text, fused_action = self.fuse_embeddings(image, instruction, possible_action_histories)
        
        # Compute cosine similarity scores
        scores = torch.matmul(fused_image_text, fused_action.T).diagonal()
        scores = scores.cpu().numpy()
        
        predicted_idx = scores.argmax()
        predicted_history = possible_action_histories[predicted_idx]
        
        history_scores = {str(i): float(scores[i]) for i in range(len(scores))}
        return predicted_history, history_scores


def sample_and_test_bridge_efficient_ensemble(bridge_dataset_dict, model_paths, history_length,
                                               backbone='hf-hub:timm/ViT-L-16-SigLIP2-384',
                                               use_transformer=False, num_samples=10, 
                                               action_pool_size=20, images_folder=None):
    """
    Sample and test on Bridge dataset with efficient ensemble
    
    Args:
        bridge_dataset_dict: Dictionary loaded from bridge dataset JSON
        model_paths: List of paths to model checkpoints
        history_length: Action history length
        backbone: SigLIP2 model backbone (shared)
        use_transformer: Whether models use transformer encoder
        num_samples: Number of samples to test
        action_pool_size: Size of action pool for each test
        images_folder: Path to images folder
    """
    inference_model = EfficientVLA_SigLIP_Ensemble_Bridge(
        model_paths,
        history_length=history_length,
        backbone=backbone,
        use_transformer=use_transformer
    )
    
    # Extract samples from bridge dataset format
    action_histories = bridge_dataset_dict['action_histories']
    instructions = bridge_dataset_dict['instructions']
    samples = bridge_dataset_dict['samples']
    
    print("\nProcessing bridge dataset samples...")
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
            image_path = os.path.join(os.path.dirname(model_paths[0]), '../../10episodes_imgs', agent_view_image_file)
        
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
    
    for idx in tqdm(sampled_indices, desc="Testing samples with efficient ensemble"):
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
        
        # Make prediction with efficient ensemble
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
    """Display the results of the evaluation"""
    correct = 0
    ranks = []
    l2_distances = []

    print("\n--- Efficient Ensemble Evaluation Results ---")
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
    parser = argparse.ArgumentParser(description='Efficient VLA-SigLIP Ensemble (Shared Encoder)')
    
    # Model configurations
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='Paths to model checkpoints (space-separated)')
    parser.add_argument('--backbone', type=str, default='hf-hub:timm/ViT-L-16-SigLIP2-384',
                       help='Shared SigLIP2 backbone for all models')
    parser.add_argument('--use_transformer', action='store_true',
                       help='Models use transformer encoder (if not set, use MLP)')
    parser.add_argument('--history_length', type=int, default=10,
                       help='Action history length')

    # Dataset
    parser.add_argument('--bridge_dataset', type=str, required=True,
                       help='Path to the bridge dataset .json file')
    parser.add_argument('--images_folder', type=str, required=True,
                       help='Path to folder containing agent view images')

    # Evaluation Params
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test')
    parser.add_argument('--action_pool_size', type=int, default=20,
                       help='Size of the action history pool')

    args = parser.parse_args()

    print(f"Setting up efficient ensemble with {len(args.model_paths)} model(s)")
    print(f"Shared backbone: {args.backbone}")

    # Verify all model paths exist
    for model_path in args.model_paths:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            exit(1)

    # Load bridge dataset
    print(f"\nLoading bridge dataset: {args.bridge_dataset}")
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
    print(f"\nStarting efficient ensemble evaluation...")
    print(f"History length: {args.history_length}")
    print(f"Use transformer: {args.use_transformer}")
    
    results = sample_and_test_bridge_efficient_ensemble(
        bridge_dataset_dict=dataset_dict,
        model_paths=args.model_paths,
        history_length=args.history_length,
        backbone=args.backbone,
        use_transformer=args.use_transformer,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size,
        images_folder=args.images_folder
    )

    # Display results
    display_results(results)

