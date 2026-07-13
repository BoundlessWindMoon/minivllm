#!/usr/bin/env bash
# Run the full verify + bench + NCU sweep for a single megakernel variant.
#
# Usage:
#   bash scripts/run_variant.sh <variant_key> [--ncu-full] [--steps N] [--input-len N] [--output-len N]
#
# Writes results to: log/ablation/<variant_key>/
#   verify.txt           — scripts/verify_megakernel.py output
#   bench.txt            — scripts/bench_megakernel.py output
#   ncu/profile.ncu-rep  — NCU binary report (megakernel decode kernels only)
#   ncu/profile.csv      — raw NCU CSV export
#   ncu/profile_report.txt — parse_ncu_csv.py human-readable summary
#   ncu/profile.json     — parse_ncu_csv.py JSON for collect_ablation.py
#
# Environment variable used downstream: MINI_VLLM_MK_VARIANT.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <variant_key> [--ncu-full] [--steps N] [--input-len N] [--output-len N]"
    exit 2
fi

VARIANT="$1"
shift

NCU_FULL=false
STEPS=20
INPUT_LEN=32
OUTPUT_LEN=128
NUM_WARMUP=20
NUM_RUNS=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --ncu-full)   NCU_FULL=true; shift ;;
        --steps)      STEPS="$2"; shift 2 ;;
        --input-len)  INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --num-warmup) NUM_WARMUP="$2"; shift 2 ;;
        --num-runs)   NUM_RUNS="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUT_DIR="$PROJECT_DIR/log/ablation/$VARIANT"
NCU_DIR="$OUT_DIR/ncu"
mkdir -p "$NCU_DIR"

export MINI_VLLM_MK_VARIANT="$VARIANT"
export TOKENIZERS_PARALLELISM=false
# NCU-only env: the model runner brackets the decode loop with cudaProfilerStart/Stop
# (see engine/model_runner.py); ncu --profile-from-start off skips prefill.
unset MINI_VLLM_NCU_DECODE

echo "==> [$VARIANT] verify_megakernel.py --steps $STEPS"
python "$PROJECT_DIR/scripts/verify_megakernel.py" --steps "$STEPS" \
    > "$OUT_DIR/verify.txt" 2>&1 || {
        echo "    verify FAILED — see $OUT_DIR/verify.txt"
        tail -n 20 "$OUT_DIR/verify.txt"
        exit 1
    }
grep -E "(PASS|FAIL|All tokens|Max logit|Min cos)" "$OUT_DIR/verify.txt" || true

echo "==> [$VARIANT] bench_megakernel.py (--backend megakernel)"
python "$PROJECT_DIR/scripts/bench_megakernel.py" \
    --backend megakernel \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-warmup "$NUM_WARMUP" \
    --num-runs "$NUM_RUNS" \
    > "$OUT_DIR/bench.txt" 2>&1 || {
        echo "    bench FAILED — see $OUT_DIR/bench.txt"
        tail -n 30 "$OUT_DIR/bench.txt"
        exit 1
    }
grep -E "(megakernel|decode|tok/s|tokens/sec)" "$OUT_DIR/bench.txt" | tail -n 10 || true

echo "==> [$VARIANT] NCU profile (decode-only via cudaProfilerStart/Stop)"
NCU_FLAGS=""
if $NCU_FULL; then
    NCU_FLAGS="--set full"
    echo "    FULL NCU set – this run will take 5-30 min"
fi

export MINI_VLLM_NCU_DECODE=1
# shellcheck disable=SC2086
ncu \
    --profile-from-start off \
    --launch-count 10 \
    $NCU_FLAGS \
    --export "$NCU_DIR/profile.ncu-rep" \
    --csv \
    --page raw \
    --force-overwrite \
    python "$PROJECT_DIR/main.py" --config "$PROJECT_DIR/configs/profile.yaml" \
    > "$NCU_DIR/ncu_stdout.txt" 2>&1 || {
        echo "    NCU FAILED — see $NCU_DIR/ncu_stdout.txt"
        tail -n 30 "$NCU_DIR/ncu_stdout.txt"
        exit 1
    }
unset MINI_VLLM_NCU_DECODE

ncu --import "$NCU_DIR/profile.ncu-rep" --csv --page raw \
    --log-file "$NCU_DIR/profile.csv" --force-overwrite \
    > "$NCU_DIR/ncu_export.txt" 2>&1

echo "==> [$VARIANT] done. Outputs in $OUT_DIR/"
