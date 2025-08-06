"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 512, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(
            self.tokenizer.vocab_size - 1000 - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action),
                         a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        return list(self.tokenizer.vocab_size - 1000 - discretized_action)
        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    @property
    def vocab_size(self) -> int:
        return self.n_bins


# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("zhiqings/LLaVA-RLHF-13b-v1.5-336", subfolder ="sft_model")
# #tokenizer = AutoTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
# action_tokenizer = ActionTokenizer(tokenizer)
# action1 = np.array([0.0005175591257113865, -0.0016432667143329027, 0.004185314312604579, -0.006698970350627563, -0.01664876880048165, -0.005566508426010671, 0.0])
# action2 = np.array([0.006793868144893786, -0.019651793629798762, 0.003374671807119146, -0.018341272918726517, 0.047913162388673734, 0.08149821793057914, 1.0])
# print(action_tokenizer(action1))
# print(action_tokenizer(action2))
