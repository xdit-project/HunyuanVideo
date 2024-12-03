#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --prompt "a cat is running, realistic." \
    --embedded-cfg-scale 6.0 \
    --infer-steps 30 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results \
