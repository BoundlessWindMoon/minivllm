#!/usr/bin/env bash
# test/run_tests.sh  -- mini-vllm functional test runner
#
# Usage:
#   ./test/run_tests.sh               # run all tests
#   ./test/run_tests.sh unit          # unit tests only  (no GPU / model)
#   ./test/run_tests.sh integration   # integration tests (requires model)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv if present and not already active
if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

FILTER="${1:-all}"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_green()  { printf '\033[0;32m%s\033[0m' "$*"; }
_red()    { printf '\033[0;31m%s\033[0m' "$*"; }
_bold()   { printf '\033[1m%s\033[0m' "$*"; }

# ---------------------------------------------------------------------------
# Run one pytest target; print PASS / FAIL; return exit code
# ---------------------------------------------------------------------------
run_suite() {
    local label="$1"
    local target="$2"   # path passed directly to pytest
    local log_file
    log_file="$(mktemp /tmp/mini-vllm-test-XXXXXX.log)"

    printf "  %-38s" "$label"
    if python -m pytest "$target" -q --tb=short >"$log_file" 2>&1; then
        _green "PASS"
        echo
        rm -f "$log_file"
        return 0
    else
        _red "FAIL"
        echo
        sed 's/^/      /' "$log_file"
        rm -f "$log_file"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Suite table
# ---------------------------------------------------------------------------
UNIT_SUITES=(
    "unit/scheduler    |test/unit/test_scheduler.py"
    "unit/sampler      |test/unit/test_sampler.py"
    "unit/config       |test/unit/test_config.py"
    "unit/kv_pool      |test/unit/test_kv_pool.py"
    "unit/attention    |test/unit/test_attention.py"
    "unit/kivi         |test/unit/test_kivi.py"
)

INTEGRATION_SUITES=(
    "integration/single_request|test/integration/test_single_request.py"
    "integration/batch          |test/integration/test_batch.py"
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0
FAIL_LABELS=()

run_group() {
    local group_label="$1"
    local -n suites_ref="$2"

    echo
    _bold "── $group_label ──────────────────────────────────────"
    echo

    for entry in "${suites_ref[@]}"; do
        local label target
        label="$(echo "${entry%%|*}" | xargs)"
        target="$(echo "${entry##*|}" | xargs)"

        if run_suite "$label" "$target"; then
            (( PASS_COUNT++ )) || true
        else
            (( FAIL_COUNT++ )) || true
            FAIL_LABELS+=("$label")
        fi
    done
}

echo
_bold "┌──────────────────────────────────────────────────┐"
echo
_bold "  mini-vllm test runner"
echo
_bold "└──────────────────────────────────────────────────┘"

case "$FILTER" in
    unit)        run_group "Unit  (no GPU / model)"         UNIT_SUITES ;;
    integration) run_group "Integration  (requires model)"  INTEGRATION_SUITES ;;
    *)           run_group "Unit  (no GPU / model)"         UNIT_SUITES
                 run_group "Integration  (requires model)"  INTEGRATION_SUITES ;;
esac

echo
_bold "══════════════════════════════════════════════════════"
echo
TOTAL=$(( PASS_COUNT + FAIL_COUNT ))
printf "  Suites: %d   " "$TOTAL"
_green "${PASS_COUNT} passed"
printf "   "
[[ $FAIL_COUNT -gt 0 ]] && _red "${FAIL_COUNT} failed" || printf "0 failed"
echo; echo

if [[ $FAIL_COUNT -gt 0 ]]; then
    _red "  Failed:"
    echo
    for lbl in "${FAIL_LABELS[@]}"; do printf "    - %s\n" "$lbl"; done
    echo
    exit 1
else
    _green "  All suites passed."
    echo
    exit 0
fi
