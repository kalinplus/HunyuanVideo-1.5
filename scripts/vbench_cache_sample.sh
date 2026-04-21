#!/bin/bash
# VBench batch sampling with cache acceleration for HunyuanVideo-1.5
#
# Usage:
#   bash scripts/vbench_cache_sample.sh                          # all experiments
#   bash scripts/vbench_cache_sample.sh --dry-run                # print commands only
#   bash scripts/vbench_cache_sample.sh --start 0 --end 200      # prompt subset
#   bash scripts/vbench_cache_sample.sh --gpus 2                 # split across 2 GPUs

set -euo pipefail

# ============ Config ============
MODEL_ROOT='/mnt/data0/tencent/HunyuanVideo-1.5'
RESOLUTION='480p'
ASPECT_RATIO='3:4'
VIDEO_LENGTH=65
SEED=42
NUM_VIDEOS_PER_PROMPT=1
OUTPUT_BASE='./vbench_output_cache'
VBENCH_JSON='hyvideo/datasets/VBench_full_info.json'
TOTAL_PROMPTS=946
SAGEATTN='--use_sageattn'

# Cache common (shared by all experiments)
CACHE_START_STEP=3
CACHE_END_STEP=50
TOTAL_STEPS=50
NO_CACHE_BLOCK_ID='53'

# Experiment list: "cache_type:interval"
#   TeaCache    N=3, 5  (l ≈ 0.2, 0.4)
#   TaylorSeer  N=3, 5, 6  (O=1)
EXPERIMENTS=(
    # "teacache:3"
    # "teacache:5"
    # "taylorcache:3"
    "taylorcache:5"
    # "taylorcache:6"
)

# TaylorCache env vars (only effective when cache_type=taylorcache)
export TAYLOR_MAX_ORDER=1
export TAYLOR_LOW_FREQS_ORDER=1
export TAYLOR_HIGH_FREQS_ORDER=1
export TAYLOR_CUTOFF_RATIO=0

# ============ Parse args ============
export CUDA_VISIBLE_DEVICES='4,5,6,7'
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
            sed -n '2,7p' "$0" | sed 's/^# //' | sed 's/^#//'
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

TOTAL_EXPS=${#EXPERIMENTS[@]}

# ============ Experiment loop ============
for EXP_IDX in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r CACHE_TYPE CACHE_STEP_INTERVAL <<< "${EXPERIMENTS[$EXP_IDX]}"
    EXP_NUM=$((EXP_IDX + 1))

    OUTPUT_DIR="${OUTPUT_BASE}/${CACHE_TYPE}/interval_${CACHE_STEP_INTERVAL}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "============================= [${EXP_NUM}/${TOTAL_EXPS}]"
    echo "  $CACHE_TYPE  interval=$CACHE_STEP_INTERVAL"
    echo "============================="
    echo "  Prompts: $START_IDX - $END_IDX on $GPUS GPU(s)"
    echo "  Output:  $OUTPUT_DIR"
    if [ "$CACHE_TYPE" = 'taylorcache' ]; then
    echo "  Taylor:  order=$TAYLOR_MAX_ORDER, cutoff=$TAYLOR_CUTOFF_RATIO"
    fi
    echo "============================="

    PIDS=()
    for i in $(seq 0 $((GPUS - 1))); do
        S=$((START_IDX + i * PER_GPU))
        E=$((S + PER_GPU))
        [ "$E" -gt "$END_IDX" ] && E=$END_IDX
        [ "$S" -ge "$END_IDX" ] && break

        GPU_ID=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f$((i + 1)))
        LOG="${OUTPUT_DIR}/sample_gpu${GPU_ID}_${S}_${E}.log"

        CMD="CUDA_VISIBLE_DEVICES=$GPU_ID \
TAYLOR_MAX_ORDER=$TAYLOR_MAX_ORDER \
TAYLOR_LOW_FREQS_ORDER=$TAYLOR_LOW_FREQS_ORDER \
TAYLOR_HIGH_FREQS_ORDER=$TAYLOR_HIGH_FREQS_ORDER \
TAYLOR_CUTOFF_RATIO=$TAYLOR_CUTOFF_RATIO \
python scripts/eval_vbench_sample.py \
    --pretrained_model_root $MODEL_ROOT \
    --resolution $RESOLUTION \
    --aspect_ratio $ASPECT_RATIO \
    --video_length $VIDEO_LENGTH \
    --num_inference_steps $TOTAL_STEPS \
    --seed $SEED \
    --num_videos_per_prompt $NUM_VIDEOS_PER_PROMPT \
    --vbench_json_path $VBENCH_JSON \
    --index_start $S \
    --index_end $E \
    --output_dir $OUTPUT_DIR \
    --enable_cache \
    --cache_type $CACHE_TYPE \
    --cache_start_step $CACHE_START_STEP \
    --cache_end_step $CACHE_END_STEP \
    --total_steps $TOTAL_STEPS \
    --cache_step_interval $CACHE_STEP_INTERVAL \
    --no_cache_block_id $NO_CACHE_BLOCK_ID \
    $SAGEATTN"

        if [ "$DRY_RUN" = true ]; then
            echo "  [GPU $GPU_ID] prompts [$S, $E): $CMD"
        else
            echo "  [GPU $GPU_ID] prompts [$S, $E) -> $LOG"
            eval "$CMD" > "$LOG" 2>&1 &
            PIDS+=($!)
        fi
    done

    if [ "$DRY_RUN" = false ] && [ ${#PIDS[@]} -gt 0 ]; then
        echo "  Waiting for ${#PIDS[@]} worker(s)..."
        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                FAILED=$((FAILED + 1))
            fi
        done
        echo "  $([ $FAILED -eq 0 ] && echo 'Done.' || echo "WARNING: $FAILED worker(s) failed.")"
    fi
done

echo ""
echo "All ${TOTAL_EXPS} experiment(s) finished."
