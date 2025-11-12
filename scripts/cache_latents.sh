#!/bin/bash

accelerate launch --num_machines=1 --num_processes=8 cache_latents.py \
    --vae-arch="f8d4" \
    --vae-ckpt-path="/home/nsereyjo/workspace/REPA3D/log/ckpt/repae/sdvae/e2e-sdvae-400k.pt" \
    --vae-latents-name="e2e-sdvae" \
    --pproc-batch-size=128 \
    --data-dir="/datasets_local/nsereyjo/ImageNet" \
    --output-dir="/datasets_local/nsereyjo/sdvae_latents"