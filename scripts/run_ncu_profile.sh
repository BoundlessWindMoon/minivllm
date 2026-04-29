#!/usr/bin/env bash
# Run NCU (Nsight Compute) profiling on run_quantized.py and export CSV.
#
# Usage:
#   ./scripts/run_ncu_profile.sh [--full] [output_dir]
#
#   --full    Collect ALL available metrics (slower, ~5-10x overhead).
#             Without this flag, collects the complete "raw" page metrics.
#
# NOTE: NCU requires permission to access GPU performance counters.
# If you see ERR_NVGPUCTRPERM, enable permissions first:
#   sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

set -euo pipefail

FULL_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_MODE=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$PROJECT_DIR/log/ncu}"
mkdir -p "$OUTPUT_DIR"

NCU_REPORT="$OUTPUT_DIR/profile.ncu-rep"
NCU_CSV="$OUTPUT_DIR/profile.csv"

if $FULL_MODE; then
    NCU_PROFILE_FLAGS="--set full"
    echo "==> FULL MODE: Collecting ALL metrics (this will be slow)..."
else
    NCU_PROFILE_FLAGS=""
    echo "==> STANDARD MODE: Collecting raw page metrics..."
fi

echo "    Target: run_quantized.py"
echo "    Report: $NCU_REPORT"
echo "    CSV:    $NCU_CSV"
echo ""

# Prevent huggingface tokenizers from complaining about fork
export TOKENIZERS_PARALLELISM=false

# Run profiling
# --launch-count 2: only sample the first 2 matching kernel invocations.
#   Combined with configs/profile.yaml (max_new_tokens=2), this caps NCU
#   replay overhead. Tune up if you need more samples.
# shellcheck disable=SC2086
set +e
ncu \
    --kernel-name "_awq_gemm_kernel" \
    --launch-count 10 \
    $NCU_PROFILE_FLAGS \
    --export "$NCU_REPORT" \
    --csv \
    --page raw \
    --force-overwrite \
    python "$PROJECT_DIR/run_quantized.py" --config "$PROJECT_DIR/configs/profile.yaml"
NCU_EXIT=$?
set -e

if [[ $NCU_EXIT -ne 0 ]]; then
    echo ""
    echo "==> ERROR: NCU profiling failed (exit code $NCU_EXIT)."
    echo "    If you see ERR_NVGPUCTRPERM, run:"
    echo "        sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0"
    echo "    Then retry."
    exit $NCU_EXIT
fi

# Export CSV from report
ncu \
    --import "$NCU_REPORT" \
    --csv \
    --page raw \
    --log-file "$NCU_CSV" \
    --force-overwrite

echo ""
echo "==> Profiling complete. Parsing metrics..."

NCU_REPORT_TXT="$OUTPUT_DIR/profile_report.txt"
python "$SCRIPT_DIR/parse_ncu_csv.py" "$NCU_CSV" > "$NCU_REPORT_TXT"

echo ""
echo "==> Done."
echo "    Report (binary): $NCU_REPORT"
echo "    CSV:             $NCU_CSV"
echo "    Parsed report:   $NCU_REPORT_TXT"
