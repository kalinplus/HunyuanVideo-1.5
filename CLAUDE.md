# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HunyuanVideo-1.5 is a lightweight (8.3B params) video generation model supporting text-to-video (T2V) and image-to-video (I2V). Built with a DiT (Diffusion Transformer) + 3D causal VAE + flow matching pipeline. Licensed under TENCENT HUNYUAN COMMUNITY LICENSE.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

Optional acceleration libraries (each requires separate install — see README):
- Flash Attention 2/3
- flex-block-attn (sparse attention, requires H-series GPUs)
- SageAttention (mutually exclusive with flex-block-attn when enabled)
- sgl-kernel==0.3.18 (fp8 gemm)

### Inference
```bash
# Basic T2V generation (8 GPUs, single GPU also works)
torchrun --nproc_per_node=8 generate.py \
  --prompt "A cat playing" \
  --resolution 480p \
  --model_path ./ckpts

# I2V: add --image_path /path/to/image.png
# Step-distilled fast mode (480p I2V): add --enable_step_distill --num_inference_steps 8
# CFG-distilled fast mode: add --cfg_distilled
# Sparse attention (720p only): add --sparse_attn
# Disable prompt rewrite: --rewrite false
```

### Training
```bash
# Single GPU
python train.py --pretrained_model_root ./ckpts

# Multi-GPU with FSDP and sequence parallelism
torchrun --nproc_per_node=8 train.py \
  --pretrained_model_root ./ckpts \
  --enable_fsdp --enable_gradient_checkpointing --sp_size 8

# LoRA fine-tuning
torchrun --nproc_per_node=8 train.py \
  --pretrained_model_root ./ckpts \
  --use_lora --lora_r 8 --lora_alpha 16 --learning_rate 1e-4

# Resume from checkpoint
python train.py --pretrained_model_root ./ckpts --resume_from_checkpoint ./outputs/checkpoint-1000
```

No linter, test suite, or CI/CD is configured in this repo.

## Architecture

### Model Pipeline
```
Text Prompt / Image
    |
Text Encoders (LLM-based + optional ByT5 for glyph-aware text rendering)
    |                         |
    |                    Vision Encoder (SigLIP, I2V only)
    |
HunyuanVideo_1_5_DiffusionTransformer (8.3B, 60 blocks)
  ├── 20x MMDoubleStreamBlock (dual-stream: image + text tokens)
  └── 40x MMSingleStreamBlock (merged single stream)
  - Hidden: 3072, Heads: 24, Patch: [1,2,2], RoPE 3D: [16,56,56]
    |
Flow Matching Scheduler (rectified flow, Euler integration)
    |
VAE Decoder (3D causal conv, spatial 8x + temporal 4x compression)
    |
Super-Resolution (optional: 480p → 720p → 1080p)
    |
Output Video
```

### Key Source Files
- `generate.py` — Inference entry point (~470 lines). Handles model loading, offloading, cache strategies, SR.
- `train.py` — Training entry point (~1280 lines). `HunyuanVideoTrainer` class with FSDP2, SP, gradient checkpointing, LoRA, Muon optimizer.
- `hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py` — Main DiT model class `HunyuanVideo_1_5_DiffusionTransformer`
- `hyvideo/models/autoencoders/hunyuanvideo_15_vae.py` — VAE (`AutoencoderKLConv3D`) with tiling support
- `hyvideo/models/text_encoders/__init__.py` — Text encoder (LLM-based, max 1000 tokens)
- `hyvideo/models/text_encoders/byT5/` — Character-level text encoder for multilingual text rendering
- `hyvideo/models/vision_encoder/` — SigLIP vision encoder for I2V
- `hyvideo/models/transformers/modules/ssta_attention.py` — SSTA (Selective & Sliding Tile Attention)
- `hyvideo/optim/muon.py` — Muon optimizer (momentum + Newton-Schulz orthogonalization)
- `hyvideo/commons/parallel_states.py` — Distributed training setup (SP, DP, FSDP2, DeviceMesh)
- `hyvideo/commons/__init__.py` — `PIPELINE_CONFIGS` dict with per-resolution/task hyperparameters
- `hyvideo/pipelines/hunyuan_video_pipeline.py` — Core inference pipeline
- `hyvideo/schedulers/scheduling_flow_match_discrete.py` — Flow matching discrete scheduler

### Configuration
No YAML configs. Uses argparse in `generate.py` and `train.py`, plus `PIPELINE_CONFIGS` dict in `hyvideo/commons/__init__.py` for default generation settings per resolution/task combination.

### Distributed Training
- **Sequence Parallelism (SP):** Splits sequence dim across GPUs (`--sp_size`, must divide world_size)
- **FSDP2:** Shards transformer blocks with DeviceMesh `[dp_replicate, fsdp_shard]`
- **Gradient Checkpointing:** Non-reentrant, selective (single-stream blocks only by default)

### Training Data Format
Dataset `__getitem__` must return:
- `pixel_values`: `torch.Tensor` — Video `[C,F,H,W]` or Image `[C,H,W]`, range `[-1,1]`
- `text`: `str`
- `data_type`: `"video"` or `"image"`
- Optional: `latents` (pre-encoded), `byt5_text_ids` + `byt5_text_mask` (pre-tokenized)
- Video temporal dim **must** be `4n+1` (e.g., 1, 5, 9, 13, 17, 21, 41, 121)

### Attention Backends
Ordered by speed: SageAttention > flex-block-attn (sparse) > Flash Attention 3 > Flash Attention 2. SageAttention auto-disables flex-block-attn. Sparse attention is only available in 720p models and requires H-series GPUs.

### Inference Optimization Flags
- `--offloading` / `--group_offloading` — CPU offloading for limited GPU memory
- `--overlap_group_offloading` — Overlaps CPU↔GPU transfers with compute (higher CPU memory usage)
- `--enable_cache --cache_type deepcache|teacache|taylorcache` — Feature caching across denoising steps
- `--cfg_distilled` — CFG-distilled model (guidance=1, ~2x speedup)
- `--enable_step_distill` — Step-distilled model (8-12 steps, ~75% speedup on RTX 4090)
