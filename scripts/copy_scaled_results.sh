#!/usr/bin/env bash
# Copy all scaled_results.json from vbench_output and vbench_output_cache
# to outputs/ while preserving the original directory structure.
# The intermediate 'vbench/' directory is stripped from the output path.
#
# Example:
#   vbench_output/origin/steps_10/vbench/scaled_results.json
#     -> outputs/vbench_output/origin/steps_10/scaled_results.json
#
# Usage:
#   bash scripts/copy_scaled_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_ROOTS=(
    "${PROJECT_ROOT}/vbench_output"
    "${PROJECT_ROOT}/vbench_output_cache"
)

OUTPUT_ROOT="${PROJECT_ROOT}/outputs"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--dry-run]"
            echo ""
            echo "Copy all scaled_results.json from vbench_output directories"
            echo "to outputs/ while preserving directory structure."
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

TOTAL=0
for src_root in "${SRC_ROOTS[@]}"; do
    if [[ ! -d "$src_root" ]]; then
        echo "Warning: source root not found, skipping: $src_root"
        continue
    fi

    while IFS= read -r file; do
        # Compute relative path from the source root
        rel_path="${file#$src_root/}"

        # Strip the intermediate 'vbench/' directory from the path
        rel_no_vbench="${rel_path//\/vbench\/scaled_results.json/\/scaled_results.json}"

        dest_file="${OUTPUT_ROOT}/${rel_no_vbench}"
        dest_dir="$(dirname "$dest_file")"

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY-RUN] $file -> $dest_file"
        else
            mkdir -p "$dest_dir"
            cp "$file" "$dest_file"
            echo "[COPY] $file -> $dest_file"
        fi

        TOTAL=$((TOTAL + 1))
    done < <(find "$src_root" -type f -name 'scaled_results.json' 2>/dev/null)
done

echo ""
echo "============================================"
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. ${TOTAL} file(s) would be copied."
else
    echo "Done. ${TOTAL} file(s) copied to ${OUTPUT_ROOT}"
fi
echo "============================================"
