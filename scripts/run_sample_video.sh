#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results
