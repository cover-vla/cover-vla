"""Materialization for RAW dataset w/ images / actions / instructions. (no tokens)"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from prismatic.vla.datasets.datasets import EpisodicRLDSDataset, RLDSDataset


@dataclass
class RLDSActionBatchTransform:
    include_images: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        batch = dict(instruction=lang, action=action, dataset_name=dataset_name)
        if self.include_images:
            batch["image"] = rlds_batch["observation"]["image_primary"][0]

        return batch


def get_vla_action_dataset(
    data_root_dir: Path,
    data_mix: str,
    default_image_resolution: Tuple[int, int, int],
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    future_action_window_size: int = 0,
    include_images: bool = True,
):
    """Only get the image / action / instruction, don't do any tokenization."""

    # TODO new batch transform
    batch_transform = RLDSActionBatchTransform(include_images=include_images)

    # Build RLDS Iterable Dataset & Return
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        # did not add support for below kwargs with episodic dataset
        future_action_window_size=future_action_window_size,
    )

    return dataset
