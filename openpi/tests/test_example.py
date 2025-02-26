import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("pi0_fast_libero_low_mem_finetune")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_libero_low_mem_finetune")

# # Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
# example = droid_policy.make_droid_example()
# result = policy.infer(example)

# # Delete the policy to free up memory.
# del policy

# print("Actions shape:", result["actions"].shape)



# config = _config.get_config("pi0_aloha_sim")

# checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
# key = jax.random.key(0)

# # Create a model from the checkpoint.
# model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

# # We can create fake observations and actions to test the model.
# obs, act = config.model.fake_obs(), config.model.fake_act()

# # Sample actions from the model.
# loss = model.compute_loss(key, obs, act)
# print("Loss shape:", loss.shape)



# # Reduce the batch size to reduce memory usage.
# config = dataclasses.replace(config, batch_size=2)

# # Load a single batch of data. This is the same data that will be used during training.
# # NOTE: In order to make this example self-contained, we are skipping the normalization step
# # since it requires the normalization statistics to be generated using `compute_norm_stats`.
# loader = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)
# obs, act = next(iter(loader))

# # Sample actions from the model.
# loss = model.compute_loss(key, obs, act)

# # Delete the model to free up memory.
# del model

# print("Loss shape:", loss.shape)
