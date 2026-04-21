#!/usr/bin/env bash
# Batch-evaluate VBench scores for all subdirectories that contain mp4 videos.
#
# Default roots are tailored for HunyuanVideo-1.5 local layout:
#   - /home/hkl/HunyuanVideo-1.5/vbench_output/origin
#   - /home/hkl/HunyuanVideo-1.5/vbench_output_cache
#
# For each discovered video directory, this script will:
#   1) run scripts/eval_vbench_calc.py in parallel across dimensions
#   2) run scripts/eval_vbench_tabulate.py to aggregate final scores
#
# Usage examples:
#   bash scripts/vbench_eval_multi_dirs.sh
#   bash scripts/vbench_eval_multi_dirs.sh --gpus 0,1,2,3
#   bash scripts/vbench_eval_multi_dirs.sh --root /path/to/videos --dry-run
#
# Notes:
# - Results are saved under each video directory: <video_dir>/vbench/
# - Logs are saved under: <video_dir>/vbench_logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python"
CALC_SCRIPT="${PROJECT_ROOT}/scripts/eval_vbench_calc.py"
TABULATE_SCRIPT="${PROJECT_ROOT}/scripts/eval_vbench_tabulate.py"
FULL_INFO_PATH="${PROJECT_ROOT}/hyvideo/datasets/VBench_full_info.json"

GPU_LIST="4,5,6,7"
DRY_RUN=false

ROOTS=(
    "/home/hkl/HunyuanVideo-1.5/vbench_output/origin"
    "/home/hkl/HunyuanVideo-1.5/vbench_output_cache"
)

print_help() {
    cat <<'EOF'
Usage: vbench_eval_multi_dirs.sh [options]

Options:
  --root PATH            Add one root to scan (repeatable).
  --clear-default-roots  Ignore built-in roots and only use --root values.
  --gpus IDS             Comma-separated GPU IDs, e.g. 0,1,2,3.
  --python-bin CMD       Python executable/command (default: python).
  --full-info PATH       Path to VBench_full_info.json.
  --dry-run              Print commands only, do not execute.
  -h, --help             Show this help.

Behavior:
  - The script scans each root recursively and finds directories that contain *.mp4.
  - For each video directory, it splits VBench 16 dimensions across provided GPUs.
  - It writes per-dimension JSON files and then generates scaled summary scores.
EOF
}

CLEAR_DEFAULT_ROOTS=false
USER_ROOTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            USER_ROOTS+=("$2")
            shift 2
            ;;
        --clear-default-roots)
            CLEAR_DEFAULT_ROOTS=true
            shift
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --full-info)
            FULL_INFO_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            print_help
            exit 1
            ;;
    esac
done

if [[ "$CLEAR_DEFAULT_ROOTS" == true ]]; then
    ROOTS=()
fi

if [[ ${#USER_ROOTS[@]} -gt 0 ]]; then
    ROOTS+=("${USER_ROOTS[@]}")
fi

if [[ ${#ROOTS[@]} -eq 0 ]]; then
    echo "Error: no roots to scan. Use --root PATH." >&2
    exit 1
fi

if [[ ! -f "$CALC_SCRIPT" ]]; then
    echo "Error: calc script not found: $CALC_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$TABULATE_SCRIPT" ]]; then
    echo "Error: tabulate script not found: $TABULATE_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$FULL_INFO_PATH" ]]; then
    echo "Error: VBench full info not found: $FULL_INFO_PATH" >&2
    exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "Error: no GPU IDs parsed from --gpus '$GPU_LIST'" >&2
    exit 1
fi

mapfile -t VIDEO_DIRS < <(
    for root in "${ROOTS[@]}"; do
        if [[ -d "$root" ]]; then
            find "$root" -mindepth 1 -type f -name '*.mp4' -printf '%h\n'
        else
            echo "Warning: root not found, skip: $root" >&2
        fi
    done | sort -u
)

if [[ ${#VIDEO_DIRS[@]} -eq 0 ]]; then
    echo "No video directories found under configured roots."
    exit 0
fi

DIM_TOTAL=16
PER_GPU=$(( (DIM_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "============================================"
echo "VBench batch evaluation"
echo "============================================"
echo "Project root:   $PROJECT_ROOT"
echo "Roots:"
for root in "${ROOTS[@]}"; do
    echo "  - $root"
done
echo "Discovered dirs: ${#VIDEO_DIRS[@]}"
echo "GPUs:           ${GPU_LIST}"
echo "Python:         ${PYTHON_BIN}"
echo "Full info:      ${FULL_INFO_PATH}"
echo "Dry run:        ${DRY_RUN}"
echo "============================================"

FAILED_DIRS=()

for video_dir in "${VIDEO_DIRS[@]}"; do
    score_dir="${video_dir}/vbench"
    log_dir="${video_dir}/vbench_logs"
    mkdir -p "$score_dir" "$log_dir"

    echo ""
    echo "--------------------------------------------"
    echo "Evaluating: $video_dir"
    echo "--------------------------------------------"

    PIDS=()
    DIM_RANGES=()

    for i in "${!GPUS[@]}"; do
        start=$(( i * PER_GPU ))
        end=$(( start + PER_GPU ))
        if [[ "$end" -gt "$DIM_TOTAL" ]]; then
            end=$DIM_TOTAL
        fi
        if [[ "$start" -ge "$DIM_TOTAL" ]]; then
            break
        fi

        gpu="${GPUS[$i]}"
        log_file="${log_dir}/calc_gpu${gpu}_${start}_${end}.log"

        cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_BIN} ${CALC_SCRIPT} \"${video_dir}\" \"${video_dir}\" --full_info_path \"${FULL_INFO_PATH}\" --device cuda:0 --start_dim ${start} --end_dim ${end}"
        DIM_RANGES+=("gpu${gpu}:${start}-${end}")

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY-RUN] $cmd"
        else
            echo "[RUN] ${cmd} > ${log_file}"
            eval "$cmd" > "$log_file" 2>&1 &
            PIDS+=("$!")
        fi
    done

    if [[ "$DRY_RUN" == false ]]; then
        calc_failed=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                calc_failed=$((calc_failed + 1))
            fi
        done

        if [[ "$calc_failed" -ne 0 ]]; then
            echo "[FAIL] ${video_dir}: ${calc_failed} calc worker(s) failed. See ${log_dir}/"
            FAILED_DIRS+=("$video_dir")
            continue
        fi

        tab_log="${log_dir}/tabulate.log"
        tab_cmd="${PYTHON_BIN} ${TABULATE_SCRIPT} --score_dir \"${score_dir}\""
        echo "[RUN] ${tab_cmd} > ${tab_log}"
        if ! eval "$tab_cmd" > "$tab_log" 2>&1; then
            echo "[FAIL] ${video_dir}: tabulate failed. See ${tab_log}"
            FAILED_DIRS+=("$video_dir")
            continue
        fi

        scaled_json="${score_dir}/scaled_results.json"
        if [[ -f "$scaled_json" ]]; then
            echo "[OK] ${video_dir}: ${scaled_json}"
        else
            echo "[WARN] ${video_dir}: tabulate finished but ${scaled_json} not found"
        fi
    fi
done

echo ""
echo "============================================"
echo "Finished VBench batch evaluation"
echo "============================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete."
    exit 0
fi

if [[ ${#FAILED_DIRS[@]} -eq 0 ]]; then
    echo "All directories evaluated successfully."
else
    echo "Failed directories (${#FAILED_DIRS[@]}):"
    for d in "${FAILED_DIRS[@]}"; do
        echo "  - $d"
    done
    exit 1
fi
