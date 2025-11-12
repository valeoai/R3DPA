import argparse
import copy
import logging
import os
import json
import math
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from dataset_kitti import KITTI360
from loss.losses import ReconstructionLoss_Single_Stage
from models.autoencoder import vae_models
from utils import count_trainable_params

logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger




#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.exp_name)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Load VAE and create a EMA of the VAE
    vae = vae_models[args.vae]().to(device)
    if args.vae_ckpt is not None:
        vae_ckpt = torch.load(args.vae_ckpt, map_location=device,weights_only=False)
        vae.load_state_dict(vae_ckpt, strict=False)   # We may not have the projection layer in the VAE
        del vae_ckpt

    print("Total trainable params in VAE:", count_trainable_params(vae))


    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(args.exp_name, config=tracker_config)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define the optimizers for VAE and VAE loss function separately

    optimizer_vae = torch.optim.AdamW(
        vae.parameters(),
        lr=args.vae_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_loss_fn = torch.optim.AdamW(
        vae_loss_fn.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data
    dataset_config = OmegaConf.load(args.dataset_config)
    train_dataset = KITTI360(args.data_dir,split='train', dataset_config=dataset_config,pool=args.pool, grid_size=(4,64))
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")


    vae.eval()

    if args.disc_pretrained_ckpt is not None:
        # Load the discriminator from a pretrained checkpoint if provided
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location=device)
        vae_loss_fn.discriminator.load_state_dict(disc_ckpt)
        if accelerator.is_main_process:
            logger.info(f"Loaded discriminator from {args.disc_pretrained_ckpt}")

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{logging_dir}/checkpoints/{ckpt_name}'

        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        vae.load_state_dict(ckpt['vae'])
        vae_loss_fn.discriminator.load_state_dict(ckpt['discriminator'])
        optimizer_vae.load_state_dict(ckpt['opt_vae'])
        optimizer_loss_fn.load_state_dict(ckpt['opt_disc'])
        global_step = ckpt['steps']

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance
    if args.compile:
        vae = torch.compile(vae, backend="inductor", mode="default")
        vae_loss_fn = torch.compile(vae_loss_fn, backend="inductor", mode="default")

    vae, vae_loss_fn, optimizer_vae, optimizer_loss_fn, train_dataloader = accelerator.prepare(
        vae, vae_loss_fn, optimizer_vae, optimizer_loss_fn, train_dataloader
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # main training loop
    for epoch in range(args.epochs):

        for batch in train_dataloader:
            range_image = batch['image'].to(device)

            vae.train()
            with accelerator.accumulate([vae, vae_loss_fn]), accelerator.autocast():

                posterior, _, recon_image = vae(range_image)                

                vae_loss, vae_loss_dict = vae_loss_fn(range_image, recon_image, posterior, global_step, "generator")
                vae_loss = vae_loss.mean()

                accelerator.backward(vae_loss)
                if accelerator.sync_gradients:
                    grad_norm_vae = accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer_vae.step()
                optimizer_vae.zero_grad(set_to_none=True)

                # discriminator loss and update
                d_loss, d_loss_dict = vae_loss_fn(range_image, recon_image, posterior, global_step, "discriminator")
                d_loss = d_loss.mean()
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(vae_loss_fn.parameters(), args.max_grad_norm)
                optimizer_loss_fn.step()
                optimizer_loss_fn.zero_grad(set_to_none=True)

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "epoch": epoch,
                    "vae_loss": accelerator.gather(vae_loss).mean().detach().item(),
                    "reconstruction_loss": accelerator.gather(vae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                    "perceptual_loss": accelerator.gather(vae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                    "kl_loss": accelerator.gather(vae_loss_dict["kl_loss"].mean()).mean().detach().item(),
                    "weighted_gan_loss": accelerator.gather(vae_loss_dict["weighted_gan_loss"].mean()).mean().detach().item(),
                    "discriminator_factor": accelerator.gather(vae_loss_dict["discriminator_factor"].mean()).mean().detach().item(),
                    "gan_loss": accelerator.gather(vae_loss_dict["gan_loss"].mean()).mean().detach().item(),
                    "d_weight": accelerator.gather(vae_loss_dict["d_weight"].mean()).mean().detach().item(),
                    "grad_norm_vae": accelerator.gather(grad_norm_vae).mean().detach().item(),
                    "d_loss": accelerator.gather(d_loss).mean().detach().item(),
                    "grad_norm_disc": accelerator.gather(grad_norm_disc).mean().detach().item(),
                    "logits_real": accelerator.gather(d_loss_dict["logits_real"].mean()).mean().detach().item(),
                    "logits_fake": accelerator.gather(d_loss_dict["logits_fake"].mean()).mean().detach().item(),
                    "lecam_loss": accelerator.gather(d_loss_dict["lecam_loss"].mean()).mean().detach().item(),
                }
                progress_bar.set_postfix(**logs)
                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `vae` is wrapped by the `accelerator` object, so we need to unwrap it
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    unwrapped_vae_loss_fn = accelerator.unwrap_model(vae_loss_fn)

                    # model might be compiled, we extract the original model
                    original_vae = unwrapped_vae._orig_mod if args.compile else unwrapped_vae
                    original_discriminator = unwrapped_vae_loss_fn._orig_mod.discriminator if args.compile else unwrapped_vae_loss_fn.discriminator

                    checkpoint = {
                        "vae": original_vae.state_dict(),
                        "discriminator": original_discriminator.state_dict(),
                        "opt_vae": optimizer_vae.state_dict(),
                        "opt_disc": optimizer_loss_fn.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)

    
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

    # dataset params
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--resolution",type=int, nargs='+', default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset-config", type=str, default="configs/kitti.yaml")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "front"])

    # precision params
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)
    
    # vae params
    parser.add_argument("--vae", type=str, default="f8d4", choices=["f8d4", "f16d32", "h16d4", "h16p2d16", "h2p4d8", "h1p4d4"],)
    parser.add_argument("--vae-ckpt", type=str, default=None)

    # vae loss params
    parser.add_argument("--disc-pretrained-ckpt", type=str, default=None)
    parser.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_kl_gan.yaml")

    # vae training params
    parser.add_argument("--vae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=1e-4)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
