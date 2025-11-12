#!/bin/bash

srun accelerate launch train_ldm_only.py \
    --max-train-steps=1000000 \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --data-dir="/scratch/project/eu-25-9/datasets/ImageNet" \
    --batch-size=256 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-B/2" \
    --checkpointing-steps=50000 \
    --vae="f8d4" \
    --vae-ckpt="pretrained_weights/repae/sdvae/e2e-sdvae-400k.pt" \
    --vae-latents-name="e2e-sdvae" \
    --learning-rate=1e-4 \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=4 \
    --output-dir="logs/training" \
    --exp-name="sit-b-2-dinov2-b-enc4-ldm-only-e2e-sdvae-0.5-4m-imagenet-uncond" \
    --num-workers=32 \
    --no-compile \
    --unconditional \
    --num-classes=1 \