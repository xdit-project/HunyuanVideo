#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node=8 \
	sample_video.py --video-size 1280 720 --video-length 129 \
     	--infer-steps 50 --prompt "A cat walks on the grass, realistic style." \
    	--flow-reverse --ulysses-degree=8 --ring-degree=1 --save-path ./results

