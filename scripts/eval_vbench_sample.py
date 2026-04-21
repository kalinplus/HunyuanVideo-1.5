#!/usr/bin/env python3
"""Batch sampling script for VBench evaluation with HunyuanVideo-1.5.

Usage:
    python scripts/eval_vbench_sample.py \
        --pretrained_model_root ./ckpts \
        --resolution 480p --aspect_ratio 3:4 \
        --video_length 65 --num_inference_steps 50 \
        --seed 42 --index_start 0 --index_end 946 \
        --output_dir ./vbench_output
"""

import os
import sys

# Ensure project root is on sys.path so `hyvideo` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import einops
import imageio
from pathlib import Path
from loguru import logger


def save_video(video, path, fps=24):
    """Save video tensor (B, C, T, H, W) to mp4 file."""
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=fps)


def main():
    parser = argparse.ArgumentParser(description='VBench batch sampling with HunyuanVideo-1.5')

    # Model
    parser.add_argument('--pretrained_model_root', type=str, required=True,
                        help='Path to pretrained model directory (contains transformer/, vae/, etc.)')

    # Sampling parameters
    parser.add_argument('--resolution', type=str, default='480p', choices=['480p', '720p'])
    parser.add_argument('--aspect_ratio', type=str, default='3:4',
                        help='Aspect ratio, e.g. "3:4" for 480x640 (HxW)')
    parser.add_argument('--video_length', type=int, default=65,
                        help='Number of frames to generate (VBench standard: 65)')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_videos_per_prompt', type=int, default=1,
                        help='Number of videos to generate per prompt')
    parser.add_argument('--negative_prompt', type=str, default='',
                        help='Negative prompt')

    # VBench prompt control
    parser.add_argument('--vbench_json_path', type=str,
                        default='hyvideo/datasets/VBench_full_info.json',
                        help='Path to VBench_full_info.json')
    parser.add_argument('--index_start', type=int, default=0,
                        help='Start index of prompts (inclusive)')
    parser.add_argument('--index_end', type=int, default=-1,
                        help='End index of prompts (exclusive). -1 means all')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save generated videos')

    # Optional model variants
    parser.add_argument('--cfg_distilled', action='store_true', default=False)
    parser.add_argument('--enable_step_distill', action='store_true', default=False)
    parser.add_argument('--sparse_attn', action='store_true', default=False)
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp32'])

    # Inference optimizations
    parser.add_argument('--use_sageattn', action='store_true', default=False,
                        help='Enable SageAttention for faster inference')
    parser.add_argument('--sage_blocks_range', type=str, default='0-59',
                        help='Block range for SageAttention (default: 0-59 for all blocks)')
    parser.add_argument('--enable_cache', action='store_true', default=False,
                        help='Enable cache for transformer')
    parser.add_argument('--cache_type', type=str, default='deepcache',
                        help='Cache type: deepcache, teacache, taylorcache')
    parser.add_argument('--no_cache_block_id', type=str, default='53',
                        help='Blocks to exclude from deepcache (e.g., 0-5 or 0,1,2)')
    parser.add_argument('--cache_start_step', type=int, default=11,
                        help='Start step to skip when using cache')
    parser.add_argument('--cache_end_step', type=int, default=45,
                        help='End step to skip when using cache')
    parser.add_argument('--total_steps', type=int, default=50,
                        help='Total inference steps')
    parser.add_argument('--cache_step_interval', type=int, default=4,
                        help='Step interval to skip when using cache')
    parser.add_argument('--taylor_max_order', type=int, default=2,
                        help='Maximum Taylor expansion order (env TAYLOR_MAX_ORDER overrides)')
    parser.add_argument('--taylor_low_freqs_order', type=int, default=2,
                        help='Low-freq derivative order (env TAYLOR_LOW_FREQS_ORDER overrides)')
    parser.add_argument('--taylor_high_freqs_order', type=int, default=2,
                        help='High-freq derivative order (env TAYLOR_HIGH_FREQS_ORDER overrides)')
    parser.add_argument('--taylor_cutoff_ratio', type=float, default=0.1,
                        help='FFT cutoff ratio (env TAYLOR_CUTOFF_RATIO overrides)')

    args = parser.parse_args()

    # Lazy imports after env vars are set
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.commons.infer_state import initialize_infer_state

    # Initialize parallel state (single GPU by default)
    parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    # Load VBench prompts
    vbench_json_path = args.vbench_json_path
    if not os.path.isfile(vbench_json_path):
        raise FileNotFoundError(f"VBench JSON not found: {vbench_json_path}")
    with open(vbench_json_path, 'r') as f:
        prompts_data = json.load(f)

    if args.index_end < 0 or args.index_end > len(prompts_data):
        args.index_end = len(prompts_data)
    selected_prompts = prompts_data[args.index_start:args.index_end]
    logger.info(f"Loaded {len(selected_prompts)} prompts (index [{args.index_start}, {args.index_end}))")

    # Determine transformer version
    task = 't2v'
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution, task, args.cfg_distilled, args.enable_step_distill, args.sparse_attn
    )
    logger.info(f"Transformer version: {transformer_version}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create pipeline
    transformer_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    device = torch.device('cuda')

    logger.info("Creating pipeline...")
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.pretrained_model_root,
        transformer_version=transformer_version,
        create_sr_pipeline=False,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=device,
    )

    # Apply inference optimizations (SageAttention, cache, etc.)
    infer_state = initialize_infer_state(args)
    needs_optimization = args.use_sageattn or args.enable_cache
    if needs_optimization:
        pipe.apply_infer_optimization(infer_state=infer_state)
        if args.use_sageattn:
            logger.info(f"SageAttention enabled on blocks {args.sage_blocks_range}")
        if args.enable_cache:
            logger.info(f"Cache enabled: type={args.cache_type}, "
                        f"window=[{args.cache_start_step}, {args.cache_end_step}], "
                        f"interval={args.cache_step_interval}")
            if args.cache_type == 'taylorcache':
                logger.info(f"  TaylorCache params: max_order={infer_state.taylor_max_order}, "
                            f"low_freqs={infer_state.taylor_low_freqs_order}, "
                            f"high_freqs={infer_state.taylor_high_freqs_order}, "
                            f"cutoff_ratio={infer_state.taylor_cutoff_ratio}")

    # Batch generation
    total = len(selected_prompts)
    failed = []

    for idx, item in enumerate(selected_prompts):
        prompt_text = item.get("prompt_en", "")
        logger.info(f"[{idx + 1}/{total}] Generating: {prompt_text[:80]}...")

        for seed_offset in range(args.num_videos_per_prompt):
            current_seed = args.seed + seed_offset
            # VBench compatible naming: {prompt_text}-{seed_offset}.mp4
            save_path = os.path.join(args.output_dir, f"{prompt_text}-{seed_offset}.mp4")

            # Skip if already exists
            if os.path.exists(save_path):
                logger.info(f"  Skipping (exists): {save_path}")
                continue

            try:
                out = pipe(
                    prompt=prompt_text,
                    aspect_ratio=args.aspect_ratio,
                    video_length=args.video_length,
                    num_inference_steps=args.num_inference_steps,
                    negative_prompt=args.negative_prompt if args.negative_prompt else None,
                    seed=current_seed,
                    output_type="pt",
                    prompt_rewrite=False,
                    enable_sr=False,
                )
                save_video(out.videos, save_path)
                logger.info(f"  Saved: {save_path}")
            except Exception as e:
                logger.error(f"  FAILED [{prompt_text[:60]}...]: {e}")
                failed.append((idx, prompt_text, str(e)))

    # Summary
    logger.info(f"Done. Generated {total * args.num_videos_per_prompt - len(failed)} videos, "
                f"failed: {len(failed)}")
    if failed:
        logger.warning("Failed prompts:")
        for idx, prompt, err in failed:
            logger.warning(f"  [{idx}] {prompt[:60]}... | {err}")


if __name__ == '__main__':
    main()
