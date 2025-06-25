import torch
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
from finetune_trajectory_dino import VLA_BERT_DINO
from model import ModelConfig
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import argparse
import torch.nn.functional as F
from transformers import BertTokenizer
import pickle

ACTION_PADDING_VALUE = -5.0  # Define padding value globally

class VLA_DINO_Inference:
    def __init__(self, model_path, history_length, use_transformer=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.history_length = history_length
        self.use_transformer = use_transformer

        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_text_length = 77

        # ModelConfig for BERT+DINO
        # The action_dim and pooling dims are not needed for inference, but ModelConfig expects them
        # We'll set them to typical values (action_dim will be set after loading a sample if needed)
        self.model_config = ModelConfig(
            clip_model=None,
            history_length=history_length,
            text_pooling_output_dim=768,
            vision_pooling_output_dim=768
        )
        self.model = self._init_model(model_path, device)

        # DINOv2 image preprocessing pipeline (ImageNet normalization)
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_keys = ('agentview_rgb', 'eye_in_hand_rgb')

    def _init_model(self, model_path, device):
        model = VLA_BERT_DINO(self.model_config, use_transformer=self.use_transformer).to(device)
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        model.float()
        print("Model loaded successfully.")
        return model

    def predict(self, image_tuple, instruction, possible_action_histories):
        """
        Predict the most likely action history given a tuple of images and instruction(s).
        Args:
            image_tuple: (agentview_image, eye_in_hand_image) as PIL Images or numpy arrays
            instruction: String instruction OR list of instructions (one per action history)
            possible_action_histories: List of numpy action history arrays (Shape [H, D])
        Returns:
            predicted_history: The most likely action history (numpy array)
            history_scores: Dictionary mapping history index (str) to score (float)
        """
        img1, img2 = image_tuple
        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1.astype('uint8'))
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2.astype('uint8'))
        img1_tensor = self.preprocess(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.preprocess(img2).unsqueeze(0).to(self.device)

        # Support both single instruction and list of instructions
        if isinstance(instruction, list):
            # List of instructions, one per action history
            assert len(instruction) == len(possible_action_histories), "Number of instructions must match number of action histories."
            tokenized = self.tokenizer(
                instruction,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
                return_tensors='pt'
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            num_histories = len(possible_action_histories)
            # No repeat needed, already batched
        else:
            # Single instruction, repeat for all action histories
            tokenized = self.tokenizer(
                instruction,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
                return_tensors='pt'
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            num_histories = len(possible_action_histories)
            tokenized = {k: v.repeat(num_histories, 1) for k, v in tokenized.items()}

        with torch.no_grad():
            history_batch = torch.tensor(np.array(possible_action_histories), dtype=torch.float32).to(self.device)
            img1_batch = img1_tensor.repeat(num_histories, 1, 1, 1)
            img2_batch = img2_tensor.repeat(num_histories, 1, 1, 1)
            image_logits = self.model.get_similarity_score(img1_batch, img2_batch, tokenized, history_batch)
            scores = image_logits[0, :].cpu().numpy() if image_logits.ndim == 2 else image_logits.cpu().numpy()
            predicted_idx = scores.argmax()
            predicted_history = possible_action_histories[predicted_idx]
        history_scores = {str(i): float(scores[i]) for i in range(len(scores))}
        return predicted_history, history_scores

    def get_history_score(self, image_tuple, instruction, action_history):
        """
        Calculates the VLA-BERT-DINO similarity score between a tuple of images/instruction
        and a given action history by leveraging the model's forward pass.
        Args:
            image_tuple: (agentview_image, eye_in_hand_image) as PIL Images or numpy arrays
            instruction: String instruction.
            action_history: Numpy array action history (H, D), potentially padded.
        Returns:
            score: Float similarity score tensor (on the model's device).
        """
        img1, img2 = image_tuple
        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1.astype('uint8'))
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2.astype('uint8'))
        img1_tensor = self.preprocess(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.preprocess(img2).unsqueeze(0).to(self.device)
        if isinstance(instruction, str):
            tokenized = self.tokenizer(
                instruction,
                padding='max_length',
                truncation=True,
                max_length=self.max_text_length,
                return_tensors='pt'
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        else:
            tokenized = instruction
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        history_tensor = torch.tensor(action_history, dtype=torch.float32).to(self.device)
        if history_tensor.ndim == 2:
            history_tensor = history_tensor.unsqueeze(0)
        with torch.no_grad():
            image_logits = self.model.get_similarity_score(img1_tensor, img2_tensor, tokenized, history_tensor)
            score = image_logits.squeeze()
        return score

def sample_and_test(augmented_dataset_dict, model_path, history_length, use_transformer=False, num_samples=10, action_pool_size=20):
    inference_model = VLA_DINO_Inference(model_path,
                                         history_length=history_length,
                                         use_transformer=use_transformer)
    all_samples = []
    all_histories = []
    print("Flattening dataset for evaluation...")
    for instruction, data in tqdm(augmented_dataset_dict.items()):
        instruction_samples = data.get('samples', [])
        for sample_data in instruction_samples:
            images_dict = sample_data.get('images')
            pos_hist = sample_data.get('pos_action_hist')
            if images_dict is not None and pos_hist is not None:
                if 'agentview_rgb' in images_dict and 'eye_in_hand_rgb' in images_dict:
                    img1 = images_dict['agentview_rgb']
                    img2 = images_dict['eye_in_hand_rgb']
                    all_samples.append(((img1, img2), instruction, pos_hist))
                    all_histories.append(pos_hist)
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
    for idx in tqdm(sampled_indices, desc="Testing samples"):
        (img1, img2), gt_instruction, gt_pos_hist = all_samples[idx]
        action_history_pool = [gt_pos_hist]
        gt_pos_hist_added = True
        num_needed = action_pool_size - len(action_history_pool)
        if num_needed > 0:
            candidate_pool = [h for h in all_histories if not np.array_equal(h, gt_pos_hist)]
            if len(candidate_pool) > 0:
                num_to_sample = min(num_needed, len(candidate_pool))
                sampled_histories = random.sample(candidate_pool, num_to_sample)
                action_history_pool.extend(sampled_histories)
            else:
                print(f"Warning: Not enough unique histories in dataset to fill pool for sample {idx}. Pool size: {len(action_history_pool)}")
        random.shuffle(action_history_pool)
        ground_truth_idx_in_pool = None
        for i, hist in enumerate(action_history_pool):
            if np.array_equal(hist, gt_pos_hist):
                ground_truth_idx_in_pool = i
                break
        if ground_truth_idx_in_pool is None and gt_pos_hist_added:
            print(f"Warning: Ground truth positive history lost during pooling/shuffling for sample {idx}? This shouldn't happen.")
        # print ("gt_instruction", gt_instruction)
        # input()
        predicted_history, scores = inference_model.predict(
            (img1, img2), gt_instruction, action_history_pool
        )
        is_correct = np.array_equal(predicted_history, gt_pos_hist)
        results.append({
            'images': (img1, img2),
            'instruction': gt_instruction,
            'ground_truth_pos': gt_pos_hist,
            'ground_truth_idx': ground_truth_idx_in_pool,
            'prediction': predicted_history,
            'action_pool_size': len(action_history_pool),
            'scores': scores,
            'correct': is_correct,
        })
    return results

def display_results(results):
    correct = 0
    ranks = []
    l2_distances = []

    print("\n--- Evaluation Results ---")
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"  Instruction: {result['instruction']}")
        print(f"  Action pool size: {result['action_pool_size']}")
        print(f"  Ground truth (Pos Hist) index in pool: {result['ground_truth_idx']}")

        gt_str = f"Start: {result['ground_truth_pos'][0]}, End: {result['ground_truth_pos'][-1]}"
        pred_str = f"Start: {result['prediction'][0]}, End: {result['prediction'][-1]}"
        # print(f"  Ground truth action hist: {gt_str}")
        # print(f"  Predicted action hist:  {pred_str}")

        print(f"  Correct (Predicted == GT Pos): {result['correct']}")

        if isinstance(result['prediction'], np.ndarray) and isinstance(result['ground_truth_pos'], np.ndarray):
            pred_flat = result['prediction'].flatten()
            gt_flat = result['ground_truth_pos'].flatten()
            l2_dist = np.linalg.norm(pred_flat - gt_flat)
            print(f"  L2 distance (Pred vs GT Pos): {l2_dist:.4f}")
            l2_distances.append(l2_dist)

        scores = result['scores']
        try:
            scores_int_keys = {int(k): v for k, v in scores.items()}
            sorted_scores = sorted(scores_int_keys.items(), key=lambda item: item[1], reverse=True)
            print("  Top predictions (pool_index: score):")
            for pool_idx, score in sorted_scores[:5]:
                print(f"    {pool_idx}: {score:.4f}")

            if result['ground_truth_idx'] is not None:
                gt_score = scores_int_keys.get(result['ground_truth_idx'], -float('inf'))
                rank = sum(1 for score in scores_int_keys.values() if score > gt_score) + 1
                ranks.append(rank)
                print(f"  Rank of Ground Truth (Pos): {rank}")

        except ValueError:
            print("  Error processing scores for ranking (invalid format?).")

        if result['correct']:
            correct += 1

    accuracy = correct / len(results) if results else 0
    mean_rank = np.mean(ranks) if ranks else float('nan')
    mean_l2 = np.mean(l2_distances) if l2_distances else float('nan')

    print("-" * 25)
    print(f"Overall accuracy: {accuracy:.3f} ({correct}/{len(results)})")
    print(f"Mean rank of ground truth positive history: {mean_rank:.3f}")
    print(f"Mean L2 distance (Prediction vs GT Pos): {mean_l2:.4f}")
    print("-" * 25)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VLA-BERT-DINO Trajectory Model Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained trajectory model file (.pt)')
    parser.add_argument('--history_length', type=int, required=True, help='Action history length used during training (must match dataset)')
    parser.add_argument('--use_transformer', action='store_true', help='Specify if the loaded model uses a Transformer action encoder')
    parser.add_argument('--augmented_dataset', type=str, required=True, help='Path to the augmented dataset .pkl file (with pos/neg histories)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test for evaluation')
    parser.add_argument('--action_pool_size', type=int, default=50, help='Size of the action history pool (including GT) for each test sample')

    args = parser.parse_args()

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

    print(f"\nStarting evaluation (if run directly)...")
    results = sample_and_test(
        augmented_dataset_dict=dataset_dict,
        model_path=args.model_path,
        history_length=args.history_length,
        use_transformer=args.use_transformer,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size
    )

    display_results(results) 