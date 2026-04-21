#!/usr/bin/env bash
# Batch-tabulate VBench scores for directories that already have calc results.
#
# Scans configured roots recursively and finds directories containing
# _eval_results.json / _full_info.json files, then runs
# scripts/eval_vbench_tabulate.py for each one.
#
# Usage examples:
#   bash scripts/vbench_tabulate_multi_dirs.sh
#   bash scripts/vbench_tabulate_multi_dirs.sh --root /path/to/results
#
# Results are saved under each discovered directory: <dir>/scaled_results.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python"
TABULATE_SCRIPT="${PROJECT_ROOT}/scripts/eval_vbench_tabulate.py"

DRY_RUN=false

ROOTS=(
    "/home/hkl/HunyuanVideo-1.5/vbench_output/origin"
    "/home/hkl/HunyuanVideo-1.5/vbench_output_cache"
)

print_help() {
    cat <<'EOF'
Usage: vbench_tabulate_multi_dirs.sh [options]

Options:
  --root PATH            Add one root to scan (repeatable).
  --clear-default-roots  Ignore built-in roots and only use --root values.
  --python-bin CMD       Python executable/command (default: python).
  --dry-run              Print commands only, do not execute.
  -h, --help             Show this help.

Behavior:
  - Scans each root recursively for directories containing *_eval_results.json.
  - For each match, runs eval_vbench_tabulate.py --score_dir <dir>.
  - Output scaled_results.json is written directly into the discovered directory.
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
        --python-bin)
            PYTHON_BIN="$2"
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

if [[ ! -f "$TABULATE_SCRIPT" ]]; then
    echo "Error: tabulate script not found: $TABULATE_SCRIPT" >&2
    exit 1
fi

# Discover directories that already have calc results
mapfile -t SCORE_DIRS < <(
    for root in "${ROOTS[@]}"; do
        if [[ -d "$root" ]]; then
            find "$root" -mindepth 1 -type f -name '*_eval_results.json' -printf '%h\n'
        else
            echo "Warning: root not found, skip: $root" >&2
        fi
    done | sort -u
)

if [[ ${#SCORE_DIRS[@]} -eq 0 ]]; then
    echo "No directories with *_eval_results.json found under configured roots."
    exit 0
fi

echo "============================================"
echo "VBench batch tabulation"
echo "============================================"
echo "Project root:    $PROJECT_ROOT"
echo "Roots:"
for root in "${ROOTS[@]}"; do
    echo "  - $root"
done
echo "Discovered dirs: ${#SCORE_DIRS[@]}"
echo "Python:          ${PYTHON_BIN}"
echo "Dry run:         ${DRY_RUN}"
echo "============================================"

FAILED_DIRS=()

for score_dir in "${SCORE_DIRS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "Tabulating: $score_dir"
    echo "--------------------------------------------"

    cmd="${PYTHON_BIN} ${TABULATE_SCRIPT} --score_dir \"${score_dir}\""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $cmd"
    else
        echo "[RUN] $cmd"
        if eval "$cmd"; then
            scaled_json="${score_dir}/scaled_results.json"
            if [[ -f "$scaled_json" ]]; then
                echo "[OK]  $scaled_json"
            else
                echo "[WARN] tabulate finished but ${scaled_json} not found"
            fi
        else
            echo "[FAIL] tabulate failed for ${score_dir}"
            FAILED_DIRS+=("$score_dir")
        fi
    fi
done

echo ""
echo "============================================"
echo "Finished VBench batch tabulation"
echo "============================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete."
    exit 0
fi

if [[ ${#FAILED_DIRS[@]} -eq 0 ]]; then
    echo "All directories tabulated successfully."
else
    echo "Failed directories (${#FAILED_DIRS[@]}):"
    for d in "${FAILED_DIRS[@]}"; do
        echo "  - $d"
    done
    exit 1
fi
