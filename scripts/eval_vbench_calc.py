#!/usr/bin/env python3
"""VBench metric calculation for HunyuanVideo-1.5 generated videos.

Usage:
    # Evaluate all 16 dimensions on a single GPU:
    python scripts/eval_vbench_calc.py \
        --videos_path ./vbench_output \
        --output_dir ./vbench_output

    # Evaluate a subset of dimensions (for multi-GPU parallel):
    python scripts/eval_vbench_calc.py \
        --videos_path ./vbench_output \
        --output_dir ./vbench_output \
        --start_dim 0 --end_dim 8 --device cuda:0

    # Multi-GPU example (run in separate terminals):
    # GPU 0: dimensions 0-5
    python scripts/eval_vbench_calc.py --videos_path ./vbench_output --output_dir ./vbench_output --start_dim 0 --end_dim 6 --device cuda:0
    # GPU 1: dimensions 6-11
    python scripts/eval_vbench_calc.py --videos_path ./vbench_output --output_dir ./vbench_output --start_dim 6 --end_dim 12 --device cuda:1
    # GPU 2: dimensions 12-15
    python scripts/eval_vbench_calc.py --videos_path ./vbench_output --output_dir ./vbench_output --start_dim 12 --end_dim 16 --device cuda:2
"""

import argparse
import os
import time

DIMENSIONS = [
    "subject_consistency",
    "imaging_quality",
    "background_consistency",
    "motion_smoothness",
    "overall_consistency",
    "human_action",
    "multiple_objects",
    "spatial_relationship",
    "object_class",
    "color",
    "aesthetic_quality",
    "appearance_style",
    "temporal_flickering",
    "scene",
    "temporal_style",
    "dynamic_degree",
]


def main():
    parser = argparse.ArgumentParser(description='VBench metric calculation')
    parser.add_argument('videos_path', type=str,
                        help='Path to directory containing generated .mp4 videos')
    parser.add_argument('output_dir', type=str,
                        help='Path to save evaluation results')
    parser.add_argument('--full_info_path', type=str,
                        default='hyvideo/datasets/VBench_full_info.json',
                        help='Path to VBench_full_info.json')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (e.g. cuda:0, cuda:1)')
    parser.add_argument('--start_dim', type=int, default=0,
                        help='Start index of dimensions to evaluate (inclusive)')
    parser.add_argument('--end_dim', type=int, default=-1,
                        help='End index of dimensions to evaluate (exclusive). -1 means all')

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, "vbench")
    os.makedirs(output_dir, exist_ok=True)

    if args.end_dim < 0 or args.end_dim > len(DIMENSIONS):
        args.end_dim = len(DIMENSIONS)

    selected_dims = DIMENSIONS[args.start_dim:args.end_dim]
    print(f"Evaluating {len(selected_dims)} dimensions: {selected_dims}")

    import torch
    from vbench import VBench

    kwargs = {
        "imaging_quality_preprocessing_mode": "longer",
    }

    start_time = time.time()

    # NOTE: must use torch.device("cuda"), not "cpu",
    # else object_class third-party module will fail
    device = torch.device(args.device)
    my_VBench = VBench(device, args.full_info_path, output_dir)

    for dim in selected_dims:
        print(f"\nEvaluating: {dim}")
        t0 = time.time()
        my_VBench.evaluate(
            videos_path=args.videos_path,
            name=dim,
            local=True,
            read_frame=False,
            dimension_list=[dim],
            mode="vbench_standard",
            **kwargs,
        )
        print(f"  {dim} done in {time.time() - t0:.1f}s")

    print(f"\nAll done. Total runtime: {time.time() - start_time:.1f}s")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
