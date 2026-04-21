#!/usr/bin/env bash
# Evaluate VBench for a single video directory.
#
# Usage:
#   bash scripts/vbench_eval_single_dir.sh /path/to/video_dir
#   bash scripts/vbench_eval_single_dir.sh /path/to/video_dir --gpus 0,1,2,3
#   bash scripts/vbench_eval_single_dir.sh /path/to/video_dir --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python"
CALC_SCRIPT="${PROJECT_ROOT}/scripts/eval_vbench_calc.py"
TABULATE_SCRIPT="${PROJECT_ROOT}/scripts/eval_vbench_tabulate.py"
FULL_INFO_PATH="${PROJECT_ROOT}/hyvideo/datasets/VBench_full_info.json"

GPU_LIST="4,5,6,7"
DRY_RUN=false

print_help() {
    cat <<'EOF'
Usage: vbench_eval_single_dir.sh VIDEO_DIR [options]

Arguments:
  VIDEO_DIR              Path to directory containing .mp4 videos.

Options:
  --gpus IDS             Comma-separated GPU IDs (default: 4,5,6,7).
  --python-bin CMD       Python executable (default: python).
  --full-info PATH       Path to VBench_full_info.json.
  --dry-run              Print commands only, do not execute.
  -h, --help             Show this help.
EOF
}

if [[ $# -eq 0 ]]; then
    print_help
    exit 1
fi

VIDEO_DIR="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ ! -d "$VIDEO_DIR" ]]; then
    echo "Error: video directory not found: $VIDEO_DIR" >&2
    exit 1
fi

for f in "$CALC_SCRIPT" "$TABULATE_SCRIPT" "$FULL_INFO_PATH"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: file not found: $f" >&2
        exit 1
    fi
done

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "Error: no GPU IDs parsed from --gpus '$GPU_LIST'" >&2
    exit 1
fi

score_dir="${VIDEO_DIR}/vbench"
log_dir="${VIDEO_DIR}/vbench_logs"
mkdir -p "$score_dir" "$log_dir"

DIM_TOTAL=16
PER_GPU=$(( (DIM_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "============================================"
echo "VBench single-directory evaluation"
echo "============================================"
echo "Video dir:  $VIDEO_DIR"
echo "GPUs:       $GPU_LIST"
echo "Python:     $PYTHON_BIN"
echo "Full info:  $FULL_INFO_PATH"
echo "Dry run:    $DRY_RUN"
echo "============================================"

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

    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_BIN} ${CALC_SCRIPT} \"${VIDEO_DIR}\" \"${VIDEO_DIR}\" --full_info_path \"${FULL_INFO_PATH}\" --device cuda:0 --start_dim ${start} --end_dim ${end}"
    DIM_RANGES+=("gpu${gpu}:${start}-${end}")

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $cmd"
    else
        echo "[RUN] ${cmd} > ${log_file}"
        eval "$cmd" > "$log_file" 2>&1 &
        PIDS+=("$!")
    fi
done

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete."
    exit 0
fi

calc_failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        calc_failed=$((calc_failed + 1))
    fi
done

if [[ "$calc_failed" -ne 0 ]]; then
    echo "[FAIL] ${calc_failed} calc worker(s) failed. See ${log_dir}/"
    exit 1
fi

tab_log="${log_dir}/tabulate.log"
tab_cmd="${PYTHON_BIN} ${TABULATE_SCRIPT} --score_dir \"${score_dir}\""
echo "[RUN] ${tab_cmd} > ${tab_log}"
if ! eval "$tab_cmd" > "$tab_log" 2>&1; then
    echo "[FAIL] tabulate failed. See ${tab_log}"
    exit 1
fi

scaled_json="${score_dir}/scaled_results.json"
if [[ -f "$scaled_json" ]]; then
    echo "[OK] Results: ${scaled_json}"
else
    echo "[WARN] tabulate finished but ${scaled_json} not found"
fi
