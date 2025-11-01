"""
Copied from Allen's open-pi-zero: https://github.com/allenzren/open-pi-zero 
From: https://github.com/octo-models/octo/blob/main/examples/06_pytorch_oxe_dataloader.py

This example shows how to use the `src.data` dataloader with PyTorch by wrapping it in a simple PyTorch dataloader. The config below also happens to be our exact pretraining config (except for the batch size and shuffle buffer size, which are reduced for demonstration purposes).
"""

import tensorflow as tf
import torch
import numpy as np
tf.config.set_visible_devices([], "GPU")


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        # commputed bsed on sampling weights
        self.split_transition_length, self.transition_lengths = self.__get_sampling_length()

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        # TODO(allenzren): account for sample weights?
        return self._rlds_dataset.true_total_length
        # lengths = np.array(
        #     [
        #         stats["num_transitions"]
        #         for stats in self._rlds_dataset.dataset_statistics.values()
        #     ],
        #     dtype=float,
        # )
        # if hasattr(self._rlds_dataset, "sample_weights"):
        #     lengths *= self._rlds_dataset.sample_weights
        # total_len = lengths.sum()
        # if self._is_train:
        #     return int(0.95 * total_len)
        # else:
        #     return int(0.05 * total_len)
        
    def __get_sampling_length(self):
        # this function returns the sampling
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics.values()
            ],
            dtype=float,
        )
        
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= self._rlds_dataset.sample_weights
        total_len = lengths.sum()
        if self._is_train:
            split_len = int(0.95 * total_len)
        else:
            split_len = int(0.05 * total_len)
        return split_len, lengths
