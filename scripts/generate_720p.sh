#!/bin/bash
# HunyuanVideo-1.5 720p T2V generation script

export CUDA_VISIBLE_DEVICES='4'

model_path='/mnt/data0/tencent/HunyuanVideo-1.5'
prompt='A girl holding a paper with words "Hello, world!"'
seed=1
num_inference_steps=50
output_path='./outputs/test.mp4'

torchrun --nproc_per_node=1 generate.py \
    --prompt "$prompt" \
    --resolution 480p \
    --aspect_ratio 16:9 \
    --num_inference_steps $num_inference_steps \
    --video_length 65 \
    --model_path "$model_path" \
    --seed $seed \
    --offloading false \
    --sr false \
    --use_sageattn \
    --output_path "$output_path"
    # --enable_cache --cache_type deepcache
