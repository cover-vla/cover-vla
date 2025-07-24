# vla-comp

## Step 1: Generate Rephrases from LIBERO (no-ops) Dataset

To generate rephrased language instructions (including positive and negative rephrases), run:

```bash
cd openvla/experiments/robot/libero
python generate_libero_rephrases.py
```

- **Purpose:** This script processes the LIBERO benchmark tasks and generates multiple rephrasings for each instruction, including positive and negative variants.  
- **Configuration:** You can set the output path, number of rephrases, and task suite to process by editing the script variables at the top of `generate_libero_rephrases.py`.
- **Note:** Negative rephrases are currently not used for training, as generating random actions as negative labels is not convincing.

---

## Step 2: Generate Dataset for Verifier Training

After generating rephrases, you need to create an augmented dataset for training the trajectory verifier. At this step, the libero dataset will be loaded along with the rephrased instructions and the script will generate a dataset for training the verifier. (with each positive rephrases, we will use the same action as the original instruction). The dataset contains "action history", "wrist-image", "agent_view_image" and "language instruction".

One example is:

```bash
cd clip_verifier/scripts
python augment_dataset.py
```
Note: The script is for generating dataset for both positive and negative rephrases.(We dont use negative rephrases for now)

---

## Step 3: Train the Verifier

In this setting, we use InfoNCE loss to train the trajectory verifier, we treat only the diagnal as positive samples and the rest as negative samples. You can train the trajectory verifier using one of the provided bash scripts in `clip_verifier/bash/`:
One example is:

```bash
bash clip_verifier/bash/finetune_trajectory_transformer.sh
```


- **Purpose:** These scripts launch training for the verifier model using the specified dataset and configuration.
- **Configuration:**  
  - `--epochs`: Number of training epochs  
  - `--batch_size`: Batch size  
  - `--lr`: Learning rate  
  - `--history_length`: Length of action history  (This should be the same as the history length used for generating training data)
  - `--augmented_dataset`: Path to the `.pkl` dataset  
  - `--save_name`: Name for saving the trained model  
  - `--use_transformer`: Use transformer-based model  
  - `--use_wandb`: Enable Weights & Biases logging  
  - `--resume`: (Optional) Resume from a checkpoint

You can modify these parameters by editing the corresponding bash script or passing them directly to the Python script.

---

## Step 4: Evaluate the Verifier with VLA rollouts

Here, we rollout the VLA model with the verifier to get the trajectory scores. The proposed langauge instruction in the rollouts are pre-generated rephrases. (If its in-distribution, we use the language instructions that we generated in step 1)

Load rephrases are processed in the function: `def load_rephrases(task_suite_name):`

One example is:

```bash
cd openvla
bash run_gpu0.sh
```

Note: The script is for evaluating the VLA model with the verifier.