#!/usr/bin/env bash
# Run the full ablation sweep — every variant in dependency order.
#
# Usage:
#   bash scripts/run_ablation_all.sh [--ncu-full]
#
# Runs each variant via scripts/run_variant.sh in the dependency order:
#   naive -> p1 -> p3 -> p4 -> p6 -> p7 -> p8 -> p9 -> p10 -> p0 -> all_combined
#
# (Naive must be first because its metrics are the baseline for every doc.
# p0 is last among ablations because the original optimization_log doesn't
# specify the L2-thrashing fix mechanism — we treat it as best-effort.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

EXTRA_ARGS=()
if [[ "${1:-}" == "--ncu-full" ]]; then
    EXTRA_ARGS+=(--ncu-full)
fi

VARIANTS=(naive p1 p3 p4 p6 p7 p8 p9 p10 p0 all_combined)

for v in "${VARIANTS[@]}"; do
    echo "==================================================="
    echo "Variant: $v"
    echo "==================================================="
    bash "$SCRIPT_DIR/run_variant.sh" "$v" "${EXTRA_ARGS[@]}" || {
        echo "Variant $v failed — continuing with next."
    }
done

echo ""
echo "==> All variants run. Collecting results..."
python "$SCRIPT_DIR/collect_ablation.py" "$PROJECT_DIR/log/ablation" \
    > "$PROJECT_DIR/docs/megakernel_ablation/00_overview.md"
echo "==> Overview written to docs/megakernel_ablation/00_overview.md"
