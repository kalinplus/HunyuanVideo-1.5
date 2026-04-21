#!/usr/bin/env python3
"""VBench score aggregation and normalization for HunyuanVideo-1.5.

Reads per-dimension eval results from eval_vbench_calc.py,
applies min-max normalization, and computes Quality/Semantic/Total scores.

Usage:
    python scripts/eval_vbench_tabulate.py --score_dir ./vbench_output/vbench
"""

import argparse
import json
import os

QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1

QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

ORDERED_KEYS = [
    "total score",
    "quality score",
    "semantic score",
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "dynamic degree",
    "aesthetic quality",
    "imaging quality",
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]


def main():
    parser = argparse.ArgumentParser(description='VBench score aggregation')
    parser.add_argument('--score_dir', type=str, required=True,
                        help='Path to vbench/ directory containing _eval_results.json files')
    args = parser.parse_args()

    res_postfix = "_eval_results.json"
    info_postfix = "_full_info.json"

    files = os.listdir(args.score_dir)
    res_files = [x for x in files if res_postfix in x]
    info_files = [x for x in files if info_postfix in x]
    assert len(res_files) == len(info_files), \
        f"got {len(res_files)} res files, but {len(info_files)} info files"

    # Read raw results
    full_results = {}
    for res_file in res_files:
        # Validate info file has video list
        info_file = res_file.split(res_postfix)[0] + info_postfix
        info_path = os.path.join(args.score_dir, info_file)
        if not os.path.exists(info_path):
            print(f"Warning: missing info file {info_path}, skipping {res_file}")
            continue
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            if len(info[0].get("video_list", [])) == 0:
                print(f"Warning: {info_file} has 0 video list, skipping")
                continue

        with open(os.path.join(args.score_dir, res_file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, val in data.items():
                full_results[key] = format(val[0], ".4f")

    # Normalize scores
    scaled_results = {}
    dims = set()
    for key, val in full_results.items():
        dim = key.replace("_", " ") if "_" in key else key
        if dim not in NORMALIZE_DIC:
            print(f"Warning: unknown dimension '{dim}', skipping")
            continue
        min_val = NORMALIZE_DIC[dim]["Min"]
        max_val = NORMALIZE_DIC[dim]["Max"]
        if max_val == min_val:
            scaled_score = 0.0
        else:
            scaled_score = (float(val) - min_val) / (max_val - min_val)
        scaled_score *= DIM_WEIGHT[dim]
        scaled_results[dim] = scaled_score
        dims.add(dim)

    missing = set(NORMALIZE_DIC.keys()) - dims
    if missing:
        print(f"Warning: {len(missing)} dimensions not calculated: {missing}")

    # Compute composite scores
    quality_score = sum(scaled_results.get(d, 0) for d in QUALITY_LIST) / \
                    sum(DIM_WEIGHT[d] for d in QUALITY_LIST)
    semantic_score = sum(scaled_results.get(d, 0) for d in SEMANTIC_LIST) / \
                     sum(DIM_WEIGHT[d] for d in SEMANTIC_LIST)
    total_score = (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT) / \
                  (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    scaled_results["quality score"] = quality_score
    scaled_results["semantic score"] = semantic_score
    scaled_results["total score"] = total_score

    # Format output
    formatted_results = {"items": []}
    for key in ORDERED_KEYS:
        formatted_score = format(scaled_results[key] * 100, ".2f") + "%"
        formatted_results["items"].append({key: formatted_score})

    # Save
    all_results_path = os.path.join(args.score_dir, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(full_results, f, indent=4, sort_keys=True)
    print(f"Raw results saved to: {all_results_path}")

    scaled_results_path = os.path.join(args.score_dir, "scaled_results.json")
    with open(scaled_results_path, "w") as f:
        json.dump(formatted_results, f, indent=4, sort_keys=True)
    print(f"Scaled results saved to: {scaled_results_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"VBench Results Summary")
    print(f"{'='*50}")
    for key in ORDERED_KEYS:
        formatted_score = format(scaled_results[key] * 100, ".2f") + "%"
        print(f"  {key:<30s} {formatted_score:>8s}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
