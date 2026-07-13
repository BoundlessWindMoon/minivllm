#!/usr/bin/env bash
# Overnight policy sweep: runs skewed + mixed workloads back to back.
#
# Usage:
#   bash scripts/eval/sweep_overnight.sh
#
# Results:
#   results/sweep_skewed_<TIMESTAMP>.log
#   results/sweep_mixed_<TIMESTAMP>.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

run_sweep() {
    local config="$1"
    local label="$2"
    local log="results/sweep_${label}_${TIMESTAMP}.log"

    echo "================================================================"
    echo "Starting sweep: $label -> $log"
    echo "================================================================"
    echo "Start time: $(date)"          | tee "$log"
    echo "Config:     $config"          | tee -a "$log"
    echo "Policies:   fifo spf ljf random" | tee -a "$log"
    echo "Repeat:     5"                | tee -a "$log"
    echo "----------------------------------------------------------------" | tee -a "$log"

    python batch_main.py \
        --config "$config" \
        --sweep-policies fifo spf ljf random \
        --repeat 5 \
        2>&1 | tee -a "$log"

    echo "----------------------------------------------------------------" | tee -a "$log"
    echo "End time: $(date)" | tee -a "$log"
    echo "Done: $log"
    echo ""
}

run_sweep configs/workloads/skewed.yaml skewed
run_sweep configs/workloads/mixed.yaml  mixed

echo "All sweeps complete. Logs in results/"
