#!/bin/bash

accelerate launch train_repae.py \
    --max-train-steps=400000 \
    --allow-tf32 \
    --mixed-precision="bf16" \
    --seed=0 \
    --data-dir="dataset" \
    --output-dir="log/training" \
    --batch-size=32 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-B/2" \
    --checkpointing-steps=50000 \
    --loss-cfg-path="configs/l1_lpips_kl_gan.yaml" \
    --vae="h2p4d8" \
    --proj-coeff=0.5 \
    --encoder-depth=4 \
    --vae-align-proj-coeff=1.5 \
    --bn-momentum=0.1 \
    --exp-name="e2e-sit-b-2-h2p4d8-b32" \
    --num-classes=1 \
    --resolution 64 1024 \
    --dataset-config="configs/kitti.yaml" \
    --cfg-prob=0 \
    --adam-weight-decay=0.03 \
    --num-workers=16 \