"""Modifies TFDS dataset with a map function, updates the feature definition and stores new dataset."""

import os
import sys
from functools import partial

import tensorflow_datasets as tfds

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.oxe.preprocess.mod_functions import TFDS_MOD_FUNCTIONS
from src.data.oxe.preprocess.multithreaded_adhoc_tfds_builder import (
    MultiThreadedAdhocDatasetBuilder,
)

# avoid GCS nonsense errors
tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ["NO_GCE_CHECK"] = "true"


def mod_features(mods, features):
    """Modifies feature dict."""
    for mod in mods:
        features = TFDS_MOD_FUNCTIONS[mod].mod_features(features)
    return features


def mod_dataset_generator(builder, split, mods):
    """Modifies dataset features."""
    ds = builder.as_dataset(split=split)
    for mod in mods:
        ds = TFDS_MOD_FUNCTIONS[mod].mod_dataset(ds)
    for episode in tfds.core.dataset_utils.as_numpy(ds):
        yield episode


def main(args):
    builder = tfds.builder(args.dataset, data_dir=args.data_dir)

    features = mod_features(args.mods, builder.info.features)
    print("############# Target features: ###############")
    print(features)
    print("##############################################")
    assert args.data_dir != args.target_dir  # prevent overwriting original dataset

    mod_dataset_builder = MultiThreadedAdhocDatasetBuilder(
        name=args.dataset,
        version=builder.version,
        features=features,
        split_datasets={
            split: builder.info.splits[split] for split in builder.info.splits
        },
        config=builder.builder_config,
        data_dir=args.target_dir,
        description=builder.info.description,
        generator_fcn=partial(mod_dataset_generator, builder=builder, mods=args.mods),
        n_workers=args.n_workers,
        max_episodes_in_memory=args.max_episodes_in_memory,
    )
    mod_dataset_builder.download_and_prepare()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument(
        "--mods", type=str, nargs="+", default=["resize_and_jpeg_encode"]
    )
    parser.add_argument("--n_workers", type=int, default=10)
    parser.add_argument("--max_episodes_in_memory", type=int, default=100)
    args = parser.parse_args()

    main(args)
