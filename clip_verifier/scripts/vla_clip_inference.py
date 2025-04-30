import torch
import clip
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
from model import TextAwareVisualExtraction, ModelConfig
from finetune_trajectory import VLA_CLIP
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import argparse
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer
import pickle

ACTION_PADDING_VALUE = -5.0 # Define padding value globally

class VLA_CLIP_Inference:
    def __init__(self, model_path, history_length, use_transformer=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load the base CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        
        # --- Configuration specific to trajectory model ---
        self.trajectory_mode = True # Hardcoded for this script version
        self.history_length = history_length
        self.use_transformer = use_transformer
        # --------------------------------------------------
        
        self.tokenizer = SimpleTokenizer()
        
        # Initialize the VLA_CLIP model for trajectories
        self.model = self._init_model(model_path, device)
        self.device = device
        
        # Get CLIP's image preprocessing pipeline
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def _init_model(self, model_path, device):
        # Initialize the trajectory model
        # ModelConfig needs history_length
        model_config = ModelConfig(clip_model=self.clip_model, history_length=self.history_length)
        
        # Initialize model using the imported VLA_CLIP from finetune_trajectory
        model = VLA_CLIP(model_config, use_transformer=self.use_transformer).to(device)
        
        # Load trained weights
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()  # Set to evaluation mode
        model.float()  # Ensure model is in float32 precision
        print("Model loaded successfully.")
        return model
    
    def decode_text_embed(self, text_embed, topk=1):
        """
        Decodes optimized text embeddings to the nearest token IDs using top-k sampling.
        Cleans special tokens and stops decoding at <|endoftext|>.
        """
        with torch.no_grad():
            token_embeddings = self.clip_model.token_embedding.weight  # [vocab_size, dim]
            token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)

            text_embed = text_embed.squeeze(0)  # [seq_len, dim]
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

            decoded_token_ids = []

            for token_vec in text_embed:
                similarities = token_vec @ token_embeddings.T  # [vocab_size]
                if topk == 1:
                    token_id = similarities.argmax().item()
                else:
                    topk_vals, topk_idxs = similarities.topk(topk)
                    token_id = topk_idxs[torch.randint(0, topk, (1,)).item()].item()
                decoded_token_ids.append(token_id)

                # Stop if <|endoftext|>
                if token_id == self.tokenizer.encoder.get("<|endoftext|>"):
                    break

            # Remove <|startoftext|> if present at the beginning
            if decoded_token_ids and decoded_token_ids[0] == self.tokenizer.encoder.get("<|startoftext|>"):
                decoded_token_ids = decoded_token_ids[1:]
                
            if decoded_token_ids and decoded_token_ids[-1] == self.tokenizer.encoder.get("<|endoftext|>"):
                decoded_token_ids = decoded_token_ids[:-1]

            # Decode to string
            return self.tokenizer.decode(decoded_token_ids)


    def predict(self, image, instruction, possible_action_histories):
        """
        Predict the most likely action history given an image and instruction.

        Args:
            image: PIL Image or numpy array
            instruction: String instruction
            possible_action_histories: List of numpy action history arrays (Shape [H, D])

        Returns:
            predicted_history: The most likely action history (numpy array)
            history_scores: Dictionary mapping history index (str) to score (float)
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            # Assume it's already rotated correctly if loaded from our augmented dataset
            image = Image.fromarray(image.astype('uint8'))
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device) # Shape (1, C, H, W)

        # Tokenize instruction
        if isinstance(instruction, str):
            text_tokens = clip.tokenize(instruction).to(self.device) # Shape (1, SeqLen)
        else:
            # Assume it's already tokenized tensor
            text_tokens = instruction.to(self.device)

        with torch.no_grad():
            # Convert action histories to a batch tensor
            # List of (H, D) arrays -> Tensor (PoolSize, H, D)
            history_batch = torch.tensor(np.array(possible_action_histories), dtype=torch.float32).to(self.device)

            # The model's forward pass needs image/text repeated for each action history
            num_histories = history_batch.shape[0]
            image_batch = image_tensor.repeat(num_histories, 1, 1, 1) # Shape (PoolSize, C, H, W)
            text_batch = text_tokens.repeat(num_histories, 1)       # Shape (PoolSize, SeqLen)

            # Run inference
            # Model expects (B, H, D) actions, where B is batch size (here PoolSize)
            # It calculates (B, B) logits internally, but we only need the diagonal
            # Let's adapt the call slightly or interpret the result carefully.
            # A simpler approach: process one history at a time? No, less efficient.
            # Let's trust the model's internal contrastive calculation and extract the relevant scores.

            # Pass the batch to the model. It should handle the comparison internally.
            # The output logits_per_image will be (PoolSize, PoolSize).
            # The score for image_i vs action_j is logits_per_image[i, j]
            # Since image is repeated, image_i is the same image.
            # So, score for the single input image vs action_j is logits_per_image[0, j] (or any i)
            image_logits, action_logits = self.model(image_batch, text_batch, history_batch)

            # Extract scores of the single image against all action histories in the pool
            scores = image_logits[0, :].cpu().numpy() # Get first row, shape (PoolSize,)

            # Get predicted action history
            predicted_idx = scores.argmax()
            predicted_history = possible_action_histories[predicted_idx] # Return numpy array

        # Create a dictionary of scores for all histories in the pool
        history_scores = {str(i): float(scores[i]) for i in range(len(scores))}

        return predicted_history, history_scores

    # --- New Method for Scoring ---
    def get_history_score(self, image, instruction, action_history):
        """
        Calculates the VLA-CLIP similarity score between an image/instruction
        and a given action history.

        Args:
            image: PIL Image or numpy array (expects correct orientation).
            instruction: String instruction.
            action_history: Numpy array action history (H, D), potentially padded.

        Returns:
            score: Float similarity score.
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device) # Shape (1, C, H, W)

        # Tokenize instruction
        if isinstance(instruction, str):
            text_tokens = clip.tokenize(instruction).to(self.device) # Shape (1, SeqLen)
        else:
            text_tokens = instruction.to(self.device)

        # Prepare action history tensor
        # Input history should already be padded if necessary by the caller
        history_tensor = torch.tensor(action_history, dtype=torch.float32).to(self.device) # Shape (1, H, D)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0) # Shape (1, H, D)

        with torch.no_grad():
            # --- Get Combined Image/Text Embedding ---
            # Use the model's internal methods, assuming batch size 1
            patch_features, text_features = self.model.extract_clip_features(image_tensor, text_tokens)
            text_aware_features = self.model.text_aware_visual_extraction(patch_features, text_features) # text_features might need unsqueeze(1) if model expects (B, N, D)
            # Check model's text_aware_visual_extraction input expectation if error occurs
            if text_features.ndim == 2: text_features = text_features.unsqueeze(1) # Ensure (B, N_txt, D)
            
            vision_token = self.model.vision_poolings(text_aware_features)
            text_token = self.model.text_pooling(text_features) # Check model's text_pooling input expectation
            combined_features = torch.cat([text_token, vision_token], dim=-1)
            combined_features = self.model.input_projection(combined_features) # Shape (1, E)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            # --- End Image/Text Embedding ---

            # --- Get Action History Embedding ---
            # Re-use the forward pass logic for action encoding part
            action_histories_input = history_tensor # Shape (1, H, D)
            projected_trajectory = None
            if self.model.use_transformer:
                padding_mask = (action_histories_input[:, :, 0] == ACTION_PADDING_VALUE) # Use the defined constant
                encoded_steps = self.model.single_step_action_encoder(action_histories_input)
                encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
                transformer_output_permuted = self.model.trajectory_encoder(encoded_steps_permuted, src_key_padding_mask=padding_mask)
                transformer_output = transformer_output_permuted.permute(1, 0, 2)
                mask_expanded = (~padding_mask).unsqueeze(-1).float()
                summed_features = (transformer_output * mask_expanded).sum(dim=1)
                num_non_padded = mask_expanded.sum(dim=1)
                num_non_padded = torch.clamp(num_non_padded, min=1e-9)
                projected_trajectory = summed_features / num_non_padded
            else:
                flat_actions = action_histories_input.reshape(1, -1)
                projected_trajectory = self.model.complex_action_encoder(flat_actions)

            projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True) # Shape (1, E)
            # --- End Action History Embedding ---

            # Calculate Similarity Score (dot product)
            score = torch.matmul(combined_features, projected_trajectory.T).squeeze(0)

        return score

def sample_and_test(augmented_dataset_dict, model_path, history_length, use_transformer=False, num_samples=10, action_pool_size=20):
    """
    Sample random data points from the augmented dataset and test the trajectory model.

    Args:
        augmented_dataset_dict: Dictionary loaded from the augmented dataset .pkl file.
        model_path: Path to the trained trajectory model.
        history_length: Expected length of action histories.
        use_transformer: Whether the loaded model uses a transformer.
        num_samples: Number of samples to test.
        action_pool_size: Size of the action pool (including ground truth) for each test sample.
    """
    # Initialize the inference model
    inference_model = VLA_CLIP_Inference(model_path,
                                        history_length=history_length,
                                        use_transformer=use_transformer)

    # --- Prepare flat list of samples and all histories for pooling ---
    all_samples = []
    all_histories = [] # Collect all pos and neg histories for random pooling
    print("Flattening dataset for evaluation...")
    for instruction, data in tqdm(augmented_dataset_dict.items()):
        instruction_samples = data.get('samples', [])
        for sample_data in instruction_samples:
            image = sample_data.get('image')
            pos_hist = sample_data.get('pos_action_hist')
            neg_hist = sample_data.get('neg_action_hist')
            if image is not None and pos_hist is not None and neg_hist is not None:
                # Store the core tuple for sampling test points
                all_samples.append((image, instruction, pos_hist, neg_hist))
                # Add both histories to the global pool
                all_histories.append(pos_hist)
                all_histories.append(neg_hist)

    if not all_samples:
        print("Error: No valid samples found in the dataset.")
        return []
    if not all_histories:
        print("Error: No valid action histories found for pooling.")
        return []
    print(f"Total samples for testing: {len(all_samples)}")
    print(f"Total histories for pooling: {len(all_histories)}")

    # Randomly sample test points
    random.seed(42)  # For reproducibility
    sampled_indices = random.sample(range(len(all_samples)), min(num_samples, len(all_samples)))

    results = []
    for idx in tqdm(sampled_indices, desc="Testing samples"):
        # Get the ground truth data for this sample
        gt_image, gt_instruction, gt_pos_hist, gt_neg_hist = all_samples[idx]

        # --- Create Action History Pool ---
        action_history_pool = []
        # 1. Add the ground truth positive history
        action_history_pool.append(gt_pos_hist)
        gt_pos_hist_added = True

        # 2. Add the ground truth negative history (if pool size > 1)
        if action_pool_size > 1:
            action_history_pool.append(gt_neg_hist)

        # 3. Sample additional random histories (avoiding exact duplicates of gt_pos/gt_neg)
        num_needed = action_pool_size - len(action_history_pool)
        if num_needed > 0:
            # Create a temporary pool of candidates excluding the exact GTs for this sample
            candidate_pool = [h for h in all_histories if not np.array_equal(h, gt_pos_hist) and not np.array_equal(h, gt_neg_hist)]
            if len(candidate_pool) > 0:
                 num_to_sample = min(num_needed, len(candidate_pool))
                 sampled_histories = random.sample(candidate_pool, num_to_sample)
                 action_history_pool.extend(sampled_histories)
            else:
                 print(f"Warning: Not enough unique histories in dataset to fill pool for sample {idx}. Pool size: {len(action_history_pool)}")


        # Shuffle the pool and find the index of the ground truth *positive* history
        random.shuffle(action_history_pool)
        ground_truth_idx_in_pool = None
        for i, hist in enumerate(action_history_pool):
            if np.array_equal(hist, gt_pos_hist):
                ground_truth_idx_in_pool = i
                break
        # --- End Pool Creation ---

        if ground_truth_idx_in_pool is None and gt_pos_hist_added:
             print(f"Warning: Ground truth positive history lost during pooling/shuffling for sample {idx}? This shouldn't happen.")
             # Handle this case maybe by skipping or forcing GT in? For now, just warn.

        # Run prediction
        predicted_history, scores = inference_model.predict(
            gt_image, gt_instruction, action_history_pool
        )

        # Check if prediction matches the ground truth positive history
        is_correct = np.array_equal(predicted_history, gt_pos_hist)

        results.append({
            'image': gt_image,
            'instruction': gt_instruction,
            'ground_truth_pos': gt_pos_hist, # Store the positive GT
            'ground_truth_idx': ground_truth_idx_in_pool, # Index of POSITIVE GT in pool
            'prediction': predicted_history,
            'action_pool_size': len(action_history_pool),
            'scores': scores, # scores keyed by pool index (str)
            'correct': is_correct,
        })
    return results

def display_results(results):
    """Display the results of the sample_and_test function"""
    correct = 0
    ranks = []
    l2_distances = []

    print("\n--- Evaluation Results ---")
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Instruction: {result['instruction']}")
        print(f"  Action pool size: {result['action_pool_size']}")
        print(f"  Ground truth (Pos Hist) index in pool: {result['ground_truth_idx']}")

        # Optional: Print start/end of histories if too long
        gt_str = f"Start: {result['ground_truth_pos'][0]}, End: {result['ground_truth_pos'][-1]}"
        pred_str = f"Start: {result['prediction'][0]}, End: {result['prediction'][-1]}"
        # print(f"  Ground truth action hist: {gt_str}")
        # print(f"  Predicted action hist:  {pred_str}")

        print(f"  Correct (Predicted == GT Pos): {result['correct']}")

        # Calculate L2 distance between prediction and ground truth pos history
        if isinstance(result['prediction'], np.ndarray) and isinstance(result['ground_truth_pos'], np.ndarray):
            pred_flat = result['prediction'].flatten()
            gt_flat = result['ground_truth_pos'].flatten()
            l2_dist = np.linalg.norm(pred_flat - gt_flat)
            print(f"  L2 distance (Pred vs GT Pos): {l2_dist:.4f}")
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

            # Calculate rank of ground truth positive history
            if result['ground_truth_idx'] is not None:
                gt_score = scores_int_keys.get(result['ground_truth_idx'], -float('inf'))
                rank = sum(1 for score in scores_int_keys.values() if score > gt_score) + 1
                ranks.append(rank)
                print(f"  Rank of Ground Truth (Pos): {rank}")

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
    print(f"Mean rank of ground truth positive history: {mean_rank:.3f}")
    print(f"Mean L2 distance (Prediction vs GT Pos): {mean_l2:.4f}")
    print("-" * 25)

    # --- Optional: Display images (kept commented out) ---
    # ...

if __name__ == "__main__":
    # This block is for running vla_clip_inference.py directly for evaluation.
    
    parser = argparse.ArgumentParser(description='VLA-CLIP Trajectory Model Inference')
    # Model and Training Params
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained trajectory model file (.pt)')
    parser.add_argument('--history_length', type=int, required=True, help='Action history length used during training (must match dataset)')
    parser.add_argument('--use_transformer', action='store_true', help='Specify if the loaded model uses a Transformer action encoder')

    # Dataset
    parser.add_argument('--augmented_dataset', type=str, required=True, help='Path to the augmented dataset .pkl file (with pos/neg histories)')

    # Evaluation Params
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test for evaluation') # Increased default
    parser.add_argument('--action_pool_size', type=int, default=50, help='Size of the action history pool (including GT) for each test sample') # Increased default

    args = parser.parse_args()

    # Load augmented dataset
    print(f"Loading dataset: {args.augmented_dataset}")
    if not os.path.exists(args.augmented_dataset):
        print(f"Error: Dataset file not found at {args.augmented_dataset}")
        exit(1)
    try:
        with open(args.augmented_dataset, 'rb') as f:
            dataset_dict = pickle.load(f)
        print(f"Loaded dataset with {len(dataset_dict)} instructions.")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        exit(1)

    # Run evaluation
    print(f"\nStarting evaluation (if run directly)...")
    # Note: This direct execution might fail with relative imports.
    #       It's better to run evaluation through run_libero_eval.py or a dedicated script.
    results = sample_and_test(
        augmented_dataset_dict=dataset_dict,
        model_path=args.model_path,
        history_length=args.history_length,
        use_transformer=args.use_transformer,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size
    )

    # Display results
    display_results(results)