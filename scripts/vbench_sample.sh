#!/bin/bash
# VBench batch sampling for HunyuanVideo-1.5
#
# Usage:
#   bash scripts/eval_vbench_sample.sh                          # all 946 prompts
#   bash scripts/eval_vbench_sample.sh --dry-run                # print commands only
#   bash scripts/eval_vbench_sample.sh --start 0 --end 200      # prompt subset
#   bash scripts/eval_vbench_sample.sh --gpus 2                 # split across 2 GPUs

set -euo pipefail

# ============ Config ============
export CUDA_VISIBLE_DEVICES='4,5,6,7'

MODEL_ROOT='/mnt/data0/tencent/HunyuanVideo-1.5'
RESOLUTION='480p'
ASPECT_RATIO='3:4'
VIDEO_LENGTH=65
SEED=42
NUM_VIDEOS_PER_PROMPT=1
OUTPUT_BASE='./vbench_output'
VBENCH_JSON='hyvideo/datasets/VBench_full_info.json'
TOTAL_PROMPTS=946
SAGEATTN='--use_sageattn'

# Steps to evaluate — edit this list to add/remove configs
STEP_LIST=(50 10)

# ============ Parse args ============
GPUS=4
START_IDX=0
END_IDX=$TOTAL_PROMPTS
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)         GPUS=$2;       shift 2 ;;
        --start)        START_IDX=$2;  shift 2 ;;
        --end)          END_IDX=$2;    shift 2 ;;
        --model)        MODEL_ROOT=$2; shift 2 ;;
        --output)       OUTPUT_BASE=$2; shift 2 ;;
        --no-sageattn)  SAGEATTN='';      shift ;;
        --dry-run)      DRY_RUN=true;  shift ;;
        -h|--help)
            echo "Usage: $0 [--gpus N] [--start N] [--end N] [--model PATH] [--output DIR] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
if [ "$GPUS" -gt "$NUM_GPUS" ]; then
    echo "Error: requested $GPUS GPUs but only $NUM_GPUS in CUDA_VISIBLE_DEVICES"
    exit 1
fi

RANGE=$((END_IDX - START_IDX))
PER_GPU=$(( (RANGE + GPUS - 1) / GPUS ))

# ============ Main loop ============
for NUM_INFERENCE_STEPS in "${STEP_LIST[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/origin/steps_${NUM_INFERENCE_STEPS}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "============================="
    echo " VBench Sampling Config"
    echo "============================="
    echo "  Model:       $MODEL_ROOT"
    echo "  Resolution:  ${RESOLUTION} ${ASPECT_RATIO}"
    echo "  Frames:      $VIDEO_LENGTH"
    echo "  Steps:       $NUM_INFERENCE_STEPS"
    echo "  Seed:        $SEED"
    echo "  Prompts:     $START_IDX - $END_IDX ($RANGE total)"
    echo "  GPUs:        $GPUS"
    echo "  Output:      $OUTPUT_DIR"
    echo "  SageAttn:    ${SAGEATTN:-off}"
    echo "  Dry run:     $DRY_RUN"
    echo "============================="

    PIDS=()
    for i in $(seq 0 $((GPUS - 1))); do
        S=$((START_IDX + i * PER_GPU))
        E=$((S + PER_GPU))
        [ "$E" -gt "$END_IDX" ] && E=$END_IDX
        [ "$S" -ge "$END_IDX" ] && break

        GPU_ID=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f$((i + 1)))
        LOG="${OUTPUT_DIR}/sample_gpu${GPU_ID}_${S}_${E}.log"

        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/eval_vbench_sample.py \
            --pretrained_model_root $MODEL_ROOT \
            --resolution $RESOLUTION \
            --aspect_ratio $ASPECT_RATIO \
            --video_length $VIDEO_LENGTH \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --seed $SEED \
            --num_videos_per_prompt $NUM_VIDEOS_PER_PROMPT \
            --vbench_json_path $VBENCH_JSON \
            --index_start $S \
            --index_end $E \
            --output_dir $OUTPUT_DIR \
            $SAGEATTN"

        if [ "$DRY_RUN" = true ]; then
            echo ""
            echo "[GPU $GPU_ID] prompts [$S, $E):"
            echo "  $CMD"
        else
            echo ""
            echo "[GPU $GPU_ID] prompts [$S, $E) -> $LOG"
            eval "$CMD" > "$LOG" 2>&1 &
            PIDS+=($!)
        fi
    done

    if [ "$DRY_RUN" = false ] && [ ${#PIDS[@]} -gt 0 ]; then
        echo ""
        echo "Waiting for ${#PIDS[@]} worker(s)... (logs in $OUTPUT_DIR/sample_gpu*.log)"
        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                FAILED=$((FAILED + 1))
            fi
        done
        echo ""
        if [ "$FAILED" -eq 0 ]; then
            echo "All workers finished successfully."
        else
            echo "WARNING: $FAILED worker(s) failed. Check logs for details."
        fi
    fi
done
