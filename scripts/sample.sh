#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=1 generate.py \
    --num-fid-samples 10000 \
    --path-type linear \
    --mode sde \
    --num-steps 256 \
    --cfg-scale 1.0 \
    --guidance-high 1.0 \
    --guidance-low 0.0 \
    --exp-path log/training/tuning-e2e-sit-b-2-h2p4d8-b32 \
    --train-steps 1800000 \
    --sample-dir log/samples \
    --pproc-batch-size 32 \