#!/usr/bin/env python3
"""
Efficient ensemble inference using merged trainable components file.
Loads SigLIP encoder from HuggingFace, trainable components from merged .pt file.
"""

import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
import os
import sys
import numpy as np
import random
from tqdm import tqdm

# Use relative imports for package modules
from .model import TextAwareVisualExtraction, ModelConfig
from .finetune_trajectory_bridge_ddp import VLA_SigLIP2_Bridge
import argparse
import json


class EfficientEnsembleMerged:
    def __init__(self, merged_checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize efficient ensemble from merged checkpoint.
        Loads SigLIP encoder from HuggingFace, trainable components from file.
        
        Args:
            merged_checkpoint_path: Path to merged trainable components .pt file
            device: Device to run inference on
        """
        self.device = device
        
        print(f"Loading merged checkpoint: {os.path.basename(merged_checkpoint_path)}")
        merged_checkpoint = torch.load(merged_checkpoint_path, map_location=device)
        
        # Extract configuration
        self.backbone = merged_checkpoint['backbone']
        self.use_transformer = merged_checkpoint['use_transformer']
        self.history_length = merged_checkpoint['history_length']
        self.action_dim = merged_checkpoint['action_dim']
        self.num_models = merged_checkpoint['num_models']
        
        print(f"  Configuration:")
        print(f"    Backbone: {self.backbone}")
        print(f"    Models: {self.num_models}")
        print(f"    Use transformer: {self.use_transformer}")
        print(f"    History length: {self.history_length}")
        print(f"    Source checkpoints: {merged_checkpoint['source_checkpoints']}")
        
        # Load SigLIP encoder from HuggingFace
        print(f"\nLoading shared SigLIP encoder from HuggingFace: {self.backbone}")
        self.siglip_model, self.preprocess = create_model_from_pretrained(self.backbone)
        self.siglip_model = self.siglip_model.to(device)
        self.siglip_model.eval()
        
        # Freeze encoder
        for param in self.siglip_model.parameters():
            param.requires_grad = False
        
        # Convert to bf16 for efficiency
        self.siglip_model = self.siglip_model.to(torch.bfloat16)
        
        # Load tokenizer
        self.tokenizer = get_tokenizer(self.backbone)
        
        # Initialize one full model for extract_features method
        print(f"\nInitializing model structure...")
        model_config = ModelConfig(
            clip_model=self.siglip_model,
            history_length=self.history_length,
            action_dim=self.action_dim
        )
        self.full_model_for_features = VLA_SigLIP2_Bridge(
            model_config, 
            use_transformer=self.use_transformer
        ).to(device)
        self.full_model_for_features.eval()
        
        # Get dimensions from the model structure
        text_dim = self.siglip_model.text.output_dim
        vision_dim = self.siglip_model.visual.trunk.num_features
        visual_patch_size = self.siglip_model.visual.trunk.patch_embed.proj.kernel_size[0]
        image_size = self.siglip_model.visual.image_size[0] if hasattr(self.siglip_model.visual, 'image_size') else 224
        num_img_patches = (image_size // visual_patch_size) ** 2
        
        # Load trainable components for each model in ensemble
        print(f"\nLoading {self.num_models} trainable component sets...")
        self.trainable_models = []
        ensemble_components = merged_checkpoint['ensemble_components']
        
        for i, component_state in enumerate(ensemble_components):
            print(f"  [{i+1}/{self.num_models}] Loading trainable components...")
            
            # Create fresh modules and load state
            from .model import TextAwareVisualExtraction, AttentionPooling
            
            # Initialize modules with correct dimensions
            text_aware = TextAwareVisualExtraction(
                num_img_patches=num_img_patches,
                vision_dim=vision_dim
            ).to(device)
            text_aware.load_state_dict(component_state['text_aware_visual_extraction'])
            text_aware.eval()
            
            vision_pooling = AttentionPooling(
                input_dim=vision_dim,
                output_dim=512,
                num_heads=8,
                num_layers=4,
                num_readouts=1
            ).to(device)
            vision_pooling.load_state_dict(component_state['vision_poolings'])
            vision_pooling.eval()
            
            text_pooling = AttentionPooling(
                input_dim=text_dim,
                output_dim=512,
                num_heads=8,
                num_layers=4,
                num_readouts=1
            ).to(device)
            text_pooling.load_state_dict(component_state['text_pooling'])
            text_pooling.eval()
            
            input_projection = torch.nn.Linear(1024, 512).to(device)
            input_projection.load_state_dict(component_state['input_projection'])
            input_projection.eval()
            
            # Action encoder (transformer or MLP)
            if self.use_transformer:
                single_step_encoder = torch.nn.Linear(self.action_dim, 512).to(device)
                single_step_encoder.load_state_dict(component_state['single_step_action_encoder'])
                single_step_encoder.eval()
                
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=8,
                    dim_feedforward=1024,  # 512 * 2, matching training
                    batch_first=False,
                    dropout=0.1
                )
                trajectory_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4).to(device)
                trajectory_encoder.load_state_dict(component_state['trajectory_encoder'])
                trajectory_encoder.eval()
                
                components = {
                    'text_aware_visual_extraction': text_aware,
                    'vision_poolings': vision_pooling,
                    'text_pooling': text_pooling,
                    'input_projection': input_projection,
                    'single_step_action_encoder': single_step_encoder,
                    'trajectory_encoder': trajectory_encoder,
                    'complex_action_encoder': None,
                    'action_padding_value': component_state['action_padding_value'],
                }
            else:
                # MLP encoder matching training configuration
                complex_encoder = torch.nn.Sequential(
                    torch.nn.Linear(self.history_length * self.action_dim, 512),
                    torch.nn.LayerNorm(512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(512, 512)
                ).to(device)
                complex_encoder.load_state_dict(component_state['complex_action_encoder'])
                complex_encoder.eval()
                
                components = {
                    'text_aware_visual_extraction': text_aware,
                    'vision_poolings': vision_pooling,
                    'text_pooling': text_pooling,
                    'input_projection': input_projection,
                    'single_step_action_encoder': None,
                    'trajectory_encoder': None,
                    'complex_action_encoder': complex_encoder,
                    'action_padding_value': component_state['action_padding_value'],
                }
            
            self.trainable_models.append(components)
        
        print(f"\n✅ Successfully loaded shared encoder + {len(self.trainable_models)} trainable component sets!")
    
    def extract_shared_features(self, img_tensor, text_tokens):
        """Extract features using the shared frozen encoder"""
        with torch.no_grad():
            patch_features, text_features = self.full_model_for_features.extract_features(img_tensor, text_tokens)
            return patch_features, text_features
    
    def get_embeddings_from_model_batch(self, model_idx, patch_features, text_features, action_histories_batch):
        """
        Get embeddings using specific trainable components (batched version)
        
        Args:
            model_idx: Index of the model to use
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
            
            # Process image/text for the batch
            text_aware_features = components['text_aware_visual_extraction'](patch_features_batch, text_features_batch)
            vision_token = components['vision_poolings'](text_aware_features)
            text_token = components['text_pooling'](text_features_batch)
            
            combined_features = torch.cat([text_token, vision_token], dim=-1)
            combined_features = components['input_projection'](combined_features)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            
            # Process action histories batch
            action_histories = action_histories_batch.float()
            
            if self.use_transformer:
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
                flat_actions = action_histories.reshape(batch_size, -1)
                projected_trajectory = components['complex_action_encoder'](flat_actions)
            
            projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
            
            return combined_features, projected_trajectory
    
    def fuse_embeddings(self, image, instruction, action_histories):
        """Fuse embeddings from all models in the ensemble"""
        # Preprocess image and text
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        if isinstance(instruction, str):
            text_tokens = self.tokenizer([instruction], context_length=self.siglip_model.context_length).to(self.device)
        else:
            text_tokens = instruction.to(self.device)
            if text_tokens.ndim == 1:
                text_tokens = text_tokens.unsqueeze(0)
        
        # Extract shared features ONCE
        patch_features, text_features = self.extract_shared_features(img_tensor, text_tokens)
        
        # Convert action histories to batch tensor
        action_histories_array = np.array(action_histories)
        action_histories_batch = torch.tensor(action_histories_array, dtype=torch.float32).to(self.device)
        
        # Process through each model in batch
        all_image_text_embeds = []
        all_action_embeds = []
        
        for model_idx in range(self.num_models):
            # Process all action histories in one forward pass
            img_text_embeds, action_embeds = self.get_embeddings_from_model_batch(
                model_idx, patch_features, text_features, action_histories_batch
            )
            all_image_text_embeds.append(img_text_embeds)
            all_action_embeds.append(action_embeds)
        
        # Stack and average
        all_image_text_embeds = torch.stack(all_image_text_embeds)
        all_action_embeds = torch.stack(all_action_embeds)
        
        fused_image_text = all_image_text_embeds.mean(dim=0)
        fused_action = all_action_embeds.mean(dim=0)
        
        # Re-normalize
        fused_image_text = fused_image_text / fused_image_text.norm(dim=-1, keepdim=True)
        fused_action = fused_action / fused_action.norm(dim=-1, keepdim=True)
        
        return fused_image_text, fused_action
    
    def predict(self, image, instruction, possible_action_histories):
        """Predict the most likely action history using ensemble fusion"""
        fused_image_text, fused_action = self.fuse_embeddings(image, instruction, possible_action_histories)
        
        # Compute scores
        scores = torch.matmul(fused_image_text, fused_action.T).diagonal()
        scores = scores.cpu().numpy()
        
        predicted_idx = scores.argmax()
        predicted_history = possible_action_histories[predicted_idx]
        
        history_scores = {str(i): float(scores[i]) for i in range(len(scores))}
        return predicted_history, history_scores
    
    def compute_max_similarity_scores_batch(self, images, instructions, all_action_histories, cfg_repeat_language_instructions=1):
        """
        Compute maximum similarity scores between each (image, language) pair and all actions.
        Returns the single best action across all combinations.
        
        Args:
            images: List of PIL Images or numpy arrays
            instructions: List of instruction strings
            all_action_histories: List of ALL possible action history arrays
            
        Returns:
            max_score: Highest similarity score across all combinations
            max_instruction: Instruction that achieved the highest score
            max_action_history: Action history that achieved the highest score
            best_action_idx: Index of the best action in all_action_histories
        """
        batch_size = len(images)
        num_actions = len(all_action_histories)
        
        # print(f"Computing max similarity scores for {batch_size} (image, language) pairs against {num_actions} actions...")
        
        # OPTIMIZATION: Check if all instructions are the same (common case)
        # If so, only encode once instead of repeating
        all_same_instruction = len(set(instructions)) == 1 if isinstance(instructions[0], str) else False
        
        if all_same_instruction and batch_size > 1:
            # Optimized path: encode image and text only once
            if isinstance(images[0], np.ndarray):
                image = Image.fromarray(images[0].astype('uint8'))
            else:
                image = images[0]
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            text_tokens = self.tokenizer([instructions[0]], context_length=self.siglip_model.context_length).to(self.device)
            
            # Extract features once
            patch_features, text_features = self.extract_shared_features(img_tensor, text_tokens)
            patch_features_batch = patch_features  # (1, num_patches, dim)
            text_features_batch = text_features    # (1, num_tokens, dim)
            # Set batch_size to 1 for embedding computation
            embedding_batch_size = 1
        else:
            # Original path: encode each pair separately
            patch_features_list = []
            text_features_list = []
            
            for i in range(batch_size):
                # Preprocess image and text individually
                if isinstance(images[i], np.ndarray):
                    image = Image.fromarray(images[i].astype('uint8'))
                else:
                    image = images[i]
                img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                
                if isinstance(instructions[i], str):
                    text_tokens = self.tokenizer([instructions[i]], context_length=self.siglip_model.context_length).to(self.device)
                else:
                    text_tokens = instructions[i].to(self.device)
                    if text_tokens.ndim == 1:
                        text_tokens = text_tokens.unsqueeze(0)
                
                # Extract features for this sample
                patch_features, text_features = self.extract_shared_features(img_tensor, text_tokens)
                patch_features_list.append(patch_features)
                text_features_list.append(text_features)
            
            # Stack features into batch tensors
            patch_features_batch = torch.cat(patch_features_list, dim=0)  # (batch_size, num_patches, dim)
            text_features_batch = torch.cat(text_features_list, dim=0)    # (batch_size, num_tokens, dim)
            embedding_batch_size = batch_size
        
        # Step 3: Convert all action histories to batch tensor (pad to same length)
        max_history_len = 10
        action_histories_np = [np.array(ah) for ah in all_action_histories]
        action_dim = action_histories_np[0].shape[1] if len(action_histories_np[0].shape) > 1 else 1
        padded_action_histories = []
        for ah in action_histories_np:
            if len(ah) < max_history_len:
                padding = np.ones((max_history_len - len(ah), action_dim)) * -5
                padded_ah = np.vstack([padding, ah])
            else:
                padded_ah = ah
            padded_action_histories.append(padded_ah)
        action_histories_batch = torch.tensor(np.array(padded_action_histories), dtype=torch.float32).to(self.device)
        
        # Step 4: Process through each model in batch
        all_image_text_embeds = []
        all_action_embeds = []
        
        for model_idx in range(self.num_models):
            # Process all samples in one forward pass
            img_text_embeds, action_embeds = self.get_embeddings_from_model_batch(
                model_idx, patch_features_batch, text_features_batch, action_histories_batch
            )
            all_image_text_embeds.append(img_text_embeds)
            all_action_embeds.append(action_embeds)
        # Step 5: Stack and average across models
        all_image_text_embeds = torch.stack(all_image_text_embeds)  # (num_models, embedding_batch_size, 512)
        all_action_embeds = torch.stack(all_action_embeds)  # (num_models, num_actions, 512)
        
        fused_image_text = all_image_text_embeds.mean(dim=0)  # (embedding_batch_size, 512)
        fused_action = all_action_embeds.mean(dim=0)  # (num_actions, 512)
        # Step 6: Re-normalize
        fused_image_text = fused_image_text / fused_image_text.norm(dim=-1, keepdim=True)
        fused_action = fused_action / fused_action.norm(dim=-1, keepdim=True)
        
        # Step 7: Compute similarity matrix between all (image, language) pairs and all actions
        similarity_matrix = torch.matmul(fused_image_text, fused_action.T)  # (embedding_batch_size, num_actions)

        # --- Use the reference instruction to compare against all actions ---
        group_size = cfg_repeat_language_instructions  # e.g., 2
        num_groups = num_actions // group_size
        
        # If optimized path (embedding_batch_size=1), we already have the single row
        # Otherwise take first row (all rows are identical since instructions are all task_description)
        if embedding_batch_size == 1:
            reference_scores = similarity_matrix[0, :]  # (num_actions,) - already single comparison
        else:
            reference_scores = similarity_matrix[0, :]  # (num_actions,) - take first of identical rows
        
        # Reshape to organize by language groups
        reference_scores = reference_scores.view(num_groups, group_size)  # (num_groups, group_size)
        
        # --- Pick the best language group based on AVERAGE score ---
        avg_scores_per_language = reference_scores.mean(dim=1)  # (num_groups,) - average across actions per language
        best_group_score, best_group_idx = avg_scores_per_language.max(dim=0)  # pick language with highest avg
        
        # --- Within the selected language, pick the best action ---
        best_action_scores = reference_scores[best_group_idx]  # (group_size,) - get all action scores for selected language
        max_score, best_action_idx = best_action_scores.max(dim=0)  # pick best action within that language

        # Retrieve corresponding items
        max_score = max_score.item()
        # If all instructions are the same, just return the first one
        if all_same_instruction and batch_size > 1:
            max_instruction = instructions[0]
        else:
            max_instruction = instructions[min(best_group_idx * group_size, len(instructions) - 1)]
        
        # Map to global action index
        global_action_idx = best_group_idx * group_size + best_action_idx
        max_action_history = all_action_histories[global_action_idx]
        
        start = best_group_idx * group_size
        best_action_group = all_action_histories[start:(best_group_idx + 1) * group_size]

        # return max_score, max_instruction, max_action_history, best_action_idx, scores, best_action_group, global_action_idx
        return max_score, max_instruction, max_action_history, global_action_idx



def sample_and_test_bridge_merged_ensemble(bridge_dataset_dict, merged_checkpoint_path,
                                           num_samples=10, action_pool_size=20, images_folder=None):
    """Test using merged ensemble checkpoint"""
    inference_model = EfficientEnsembleMerged(merged_checkpoint_path)
    
    # Extract samples
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
            
        action_hist = np.array(action_histories[action_history_id])
        instruction = instructions[instruction_id]
        
        image_path = os.path.join(images_folder, agent_view_image_file)
        if not os.path.exists(image_path):
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            continue
        
        all_samples.append((image, instruction, action_hist))
        all_histories.append(action_hist)
    
    print(f"Total samples for testing: {len(all_samples)}")
    
    random.seed(42)
    sampled_indices = random.sample(range(len(all_samples)), min(num_samples, len(all_samples)))
    results = []
    
    for idx in tqdm(sampled_indices, desc="Testing samples"):
        image, gt_instruction, gt_action_hist = all_samples[idx]
        
        # Create action pool
        action_history_pool = [gt_action_hist]
        num_needed = action_pool_size - 1
        
        if num_needed > 0:
            candidate_pool = [h for h in all_histories if not np.array_equal(h, gt_action_hist)]
            if len(candidate_pool) > 0:
                num_to_sample = min(num_needed, len(candidate_pool))
                sampled_histories = random.sample(candidate_pool, num_to_sample)
                action_history_pool.extend(sampled_histories)
        
        random.shuffle(action_history_pool)
        
        # Find GT index
        ground_truth_idx = None
        for i, hist in enumerate(action_history_pool):
            if np.array_equal(hist, gt_action_hist):
                ground_truth_idx = i
                break
        
        # Predict
        predicted_history, scores = inference_model.predict(image, gt_instruction, action_history_pool)
        is_correct = np.array_equal(predicted_history, gt_action_hist)
        
        results.append({
            'instruction': gt_instruction,
            'ground_truth_action': gt_action_hist,
            'ground_truth_idx': ground_truth_idx,
            'prediction': predicted_history,
            'action_pool_size': len(action_history_pool),
            'scores': scores,
            'correct': is_correct,
        })
    
    return results


def display_results(results):
    """Display evaluation results"""
    correct = 0
    ranks = []
    l2_distances = []

    print("\n--- Efficient Ensemble Results (Merged Checkpoint) ---")
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Instruction: {result['instruction']}")
        print(f"  Correct: {result['correct']}")

        if isinstance(result['prediction'], np.ndarray) and isinstance(result['ground_truth_action'], np.ndarray):
            l2_dist = np.linalg.norm(result['prediction'].flatten() - result['ground_truth_action'].flatten())
            print(f"  L2 distance: {l2_dist:.4f}")
            l2_distances.append(l2_dist)

        scores = result['scores']
        scores_int = {int(k): v for k, v in scores.items()}
        sorted_scores = sorted(scores_int.items(), key=lambda x: x[1], reverse=True)
        print("  Top predictions:")
        for pool_idx, score in sorted_scores[:3]:
            print(f"    {pool_idx}: {score:.4f}")

        if result['ground_truth_idx'] is not None:
            gt_score = scores_int.get(result['ground_truth_idx'], -float('inf'))
            rank = sum(1 for score in scores_int.values() if score > gt_score) + 1
            ranks.append(rank)
            print(f"  Rank of GT: {rank}")

        if result['correct']:
            correct += 1

    accuracy = correct / len(results) if results else 0
    mean_rank = np.mean(ranks) if ranks else float('nan')
    mean_l2 = np.mean(l2_distances) if l2_distances else float('nan')

    print("-" * 25)
    print(f"Overall accuracy: {accuracy:.3f} ({correct}/{len(results)})")
    print(f"Mean rank: {mean_rank:.3f}")
    print(f"Mean L2 distance: {mean_l2:.4f}")
    print("-" * 25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Efficient Ensemble Inference (Merged Checkpoint)')
    
    parser.add_argument('--merged_checkpoint', type=str, required=True,
                       help='Path to merged trainable components checkpoint')
    parser.add_argument('--bridge_dataset', type=str, required=True,
                       help='Path to bridge dataset JSON')
    parser.add_argument('--images_folder', type=str, required=True,
                       help='Path to images folder')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to test')
    parser.add_argument('--action_pool_size', type=int, default=20,
                       help='Action pool size')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.merged_checkpoint):
        print(f"❌ Error: Merged checkpoint not found: {args.merged_checkpoint}")
        exit(1)
    if not os.path.exists(args.bridge_dataset):
        print(f"❌ Error: Dataset not found: {args.bridge_dataset}")
        exit(1)
    if not os.path.exists(args.images_folder):
        print(f"❌ Error: Images folder not found: {args.images_folder}")
        exit(1)
    
    # Load dataset
    with open(args.bridge_dataset, 'r') as f:
        dataset_dict = json.load(f)
    
    # Run evaluation
    print(f"\nStarting efficient ensemble evaluation (merged checkpoint)...")
    results = sample_and_test_bridge_merged_ensemble(
        bridge_dataset_dict=dataset_dict,
        merged_checkpoint_path=args.merged_checkpoint,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size,
        images_folder=args.images_folder
    )
    
    display_results(results)

