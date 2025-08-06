"""Pretrain the VQ VAE tokenizer for an arbitrary tfds dataset.

VQVAE should be installed w/ VQ-Bet instructions: https://github.com/jayLEE0301/vq_bet_official.
This script should work with arbitrary data mixtures as defined in rlds/oxe/mixtures.py.
Works with arbitrary length action chunks (see --future_action_horizon below).

---
Example with bridge for 256 bins, ac_dim=7, ac_chunk=8, and num residual rounds = 7:
     python vla-scripts/pretrain_vq.py --data_dir $WHERE_IS_BRIDGE \\
        --data_mix bridge_dataset --action_dim 7 --future_action_horizon 7 --vqvae_n_embed 256

"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import tqdm
from vqvae.vqvae import VqVae

import wandb
from prismatic.vla.action_dataset_materialize import get_vla_action_dataset


def main():
    p = argparse.ArgumentParser()

    # dataset arguments
    p.add_argument("--data_dir", type=str, required=True, help="Where to look for the tfrecords")
    p.add_argument("--data_mix", type=str, required=True, help="The name of the data [mix] to use")
    p.add_argument("--save_folder", type=str, default="vq/", help="Folder to save the final vq model (under <exp_name>)")
    p.add_argument("--shuffle_buffer_size", type=int, default=256_000)

    # train arguments
    p.add_argument("--wandb_project", type=str, default="prismatic-vq-vla")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=1028)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--save_every_n_epochs", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")

    # residual VQ arguments
    p.add_argument("--action_dim", type=int, required=True, help="Action dimension (usually 7)")
    p.add_argument("--future_action_horizon", type=int, default=9, help="How many FUTURE actions to include in VQ")
    p.add_argument("--n_latent_dims", type=int, default=512, help="Underlying VQ latent dimension")
    p.add_argument("--default_image_resolution", type=int, nargs=3, default=[3, 224, 224])
    p.add_argument(
        "--vqvae_n_embed", type=int, default=128, help="Number of token options per round, corresponds to binning width."
    )
    p.add_argument(
        "--vqvae_groups",
        type=int,
        default=None,
        help="number of residual rounds (i.e., output ac dim), defaults to ac dim",
    )
    p.add_argument("--load_dir", type=str, default=None)
    p.add_argument("--encoder_loss_multiplier", type=float, default=1.0)
    p.add_argument("--act_scale", type=float, default=1.0)

    args = p.parse_args()

    if args.vqvae_groups is None:
        args.vqvae_groups = args.action_dim

    exp_name = (
        f"pretrain_vq+mx-{args.data_mix}+fach-{args.future_action_horizon}"
        f"+ng-{args.vqvae_groups}+nemb-{args.vqvae_n_embed}+nlatent-{args.n_latent_dims}"
    )

    vla_dataset = get_vla_action_dataset(
        args.data_dir,
        args.data_mix,
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=False,
        future_action_window_size=args.future_action_horizon,
        default_image_resolution=tuple(args.default_image_resolution),
        include_images=False,
    )

    vq_config = {
        "input_dim_w": args.action_dim,  # action dimension
        # argparse fields
        "input_dim_h": args.future_action_horizon + 1,
        "n_latent_dims": args.n_latent_dims,
        "vqvae_n_embed": args.vqvae_n_embed,
        "vqvae_groups": args.vqvae_groups,
        "eval": False,
        "device": args.device,
        "load_dir": args.load_dir,
        "encoder_loss_multiplier": args.encoder_loss_multiplier,
        "act_scale": args.act_scale,
    }

    vqvae_model = VqVae(**vq_config)

    wandb.init(name=exp_name, project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # make all required directories.
    save_path = Path(args.save_folder) / exp_name
    save_path.mkdir(parents=True, exist_ok=False)
    (save_path / "checkpoints").mkdir()

    # save to experiment
    with open(save_path / "config.json", "w") as f:
        json.dump(vq_config, f, indent=4)

    train_loader = torch.utils.data.DataLoader(
        vla_dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )
    loader_iter = iter(train_loader)

    step_count = 0

    for epoch in tqdm.trange(args.epochs):
        for _ in tqdm.trange(len(train_loader)):
            batch = next(loader_iter)

            # N T D
            act = batch["action"].to(args.device)

            (
                encoder_loss,
                vq_loss_state,
                vq_code,
                vqvae_recon_loss,
            ) = vqvae_model.vqvae_update(
                act
            )  # N T D

            wandb.log({"pretrain/n_different_codes": len(torch.unique(vq_code))})
            wandb.log({"pretrain/n_different_combinations": len(torch.unique(vq_code, dim=0))})
            wandb.log({"pretrain/encoder_loss": encoder_loss})
            wandb.log({"pretrain/vq_loss_state": vq_loss_state})
            wandb.log({"pretrain/vqvae_recon_loss": vqvae_recon_loss})

            step_count += 1

        if args.save_every_n_epochs > 0 and (epoch + 1) % args.save_every_n_epochs == 0:
            print(f"Saving checkpoint after {epoch + 1} epoch(s) and {step_count} steps.")
            state_dict = vqvae_model.state_dict()
            torch.save(state_dict, os.path.join(save_path, "checkpoints/model.pt"))
            shutil.copy(
                os.path.join(save_path, "checkpoints/model.pt"),
                os.path.join(save_path, f"checkpoints/step-{step_count}-epoch-{epoch + 1}.pt"),
            )

    # SAVE AT THE END
    print("Saving last checkpoint...")
    torch.save(state_dict, os.path.join(save_path, "checkpoints/model.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
