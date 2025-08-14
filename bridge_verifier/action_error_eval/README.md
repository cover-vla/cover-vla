# Bridge Action Error Evaluation

This folder contains scripts and tools for evaluating action errors on the Bridge V2 dataset
```
## 📋 Data Structure

Each sample in `bridge_samples.json` contains:

```json
{
  "sample_id": 0,
  "state": {
    "agent_view_image_file": "0_clip.jpg",
    "timestep": 0,
    "episode_id": 0
  },
  "original_instruction": "put small spoon from basket to tray",
  "last_9_history_actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...],
  "current_ground_truth_action": [x, y, z, rx, ry, rz, gripper],
  "episode_id": 0,
  "timestep": 0
}
```

All preprocessed images are exactly 224×224 pixels.

## 🚀 Usage

### 1. Extract Bridge Dataset Samples
```bash
python extract_bridge_samples.py --num_samples 100 --builder_dir ../../bridge_dataset/1.0.0
```

### 2. Generate Instruction Rephrases
```bash
python rephrase_bridge_instructions_threaded.py
```

### 3. Generate OpenVLA Actions
```bash
python generate_vla_actions.py
```

### 4. Generate VLA-CLIP Scores
```bash
python generate_vla_clip_scores.py
```

### 5. Generate Monkey Verifier Scores
```bash
python generate_monkey_verifier_scores.py
```

### 6. Analyze and Plot Results
```bash
python plot_analysis.py
```

### 7. Preprocess Images (if needed)
```bash
python preprocessing_utils.py --image_folder bridge_images --output_folder processed_images
```

## 🔬 Analysis Pipeline

This evaluation framework follows a complete pipeline:

1. **Data Extraction**: Extract diverse samples from Bridge V2 dataset with unique instructions
2. **Instruction Rephrasing**: Generate 128 rephrases per unique instruction using LangTransform
3. **Action Generation**: Generate OpenVLA actions for all instruction variants
4. **VLA-CLIP Scoring**: Score instruction-image-action combinations using VLA-CLIP
5. **Monkey Verification**: Score using the monkey verifier for comparison
6. **Analysis**: Compare different verifier methods and plot threshold analysis

## 🔧 Dependencies

- VLA-CLIP model: `../bridge_rephrases_epoch_20.pt`
- Bridge V2 dataset: Update `--builder_dir` path as needed
- OpenVLA API: Running on `http://localhost:3200`
- Monkey Verifier API: Running on `http://127.0.0.1:3100`
