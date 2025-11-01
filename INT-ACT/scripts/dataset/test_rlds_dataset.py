import os
import sys

import draccus
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agent.configuration_pipeline import TrainPipelineConfig
from src.agent.dataset import TorchRLDSInterleavedDataset


@draccus.wrap()
def main(train_cfg: TrainPipelineConfig):
    train_dataloader = DataLoader(
                TorchRLDSInterleavedDataset(train_cfg.data.train, train=True).dataset,
                batch_size=1,
                pin_memory=True,
    )

    transition_count = 0
    for _ in train_dataloader:
        transition_count += 1
    print(f"Total usable transitions in train dataloader: {transition_count}")

if __name__ == "__main__":
    main()
