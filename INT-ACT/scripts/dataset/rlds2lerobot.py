import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "IrvingF7/taco_play_test"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "taco_play"
]  # For simplicity we will combine multiple Libero datasets into one training dataset
DATA_DIR = Path(os.environ["VLA_DATA_DIR"]) / "resize_224"

def main(data_dir: str = DATA_DIR, push_to_hub: bool = True):
    # Clean up any existing dataset in the output directory
    output_path = DATA_DIR / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=20,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        print(f"current dataset: {raw_dataset_name}")
        for episode in tqdm.tqdm(raw_dataset):
            for step in episode["steps"].as_numpy_iterator():
                step["action"] = step["action"]["rel_actions_world"]

                # clip gripper action, +1 = open, 0 = close
                step["action"] = np.concatenate(
                    (
                        step["action"][:6],
                        tf.clip_by_value(step["action"][-1:], 0, 1).numpy(),
                    )
                )
                dataset.add_frame(
                    {
                        "image": step["observation"]["rgb_static"],
                        "state": np.concatenate(
                                (
                                    step["observation"]["robot_obs"][:6],
                                    step["observation"]["robot_obs"][6:8],
                                    step["observation"]["robot_obs"][-1:],
                                ),
                            ),
                        "actions": step["action"],
                        "task": step["observation"]["natural_language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["taco_play", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main()
