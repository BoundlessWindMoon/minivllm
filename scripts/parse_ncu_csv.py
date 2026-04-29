#!/usr/bin/env python3
"""
Parse NCU (Nsight Compute) CSV output and generate a performance report.

Supports both "long" format (one metric per row) and "wide" format
(one kernel per row, metrics as columns) produced by `ncu --csv --page raw`.

This parser is metric-agnostic: it will automatically discover and report
ALL metrics present in the CSV, not just a hardcoded subset.

Usage:
    python parse_ncu_csv.py profile.csv
    python parse_ncu_csv.py profile.csv --kernel "awq_gemm" --sort time
    python parse_ncu_csv.py profile.csv --json report.json
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def _to_float(val: str) -> float:
    """Convert NCU numeric string (may contain commas, %, etc.) to float."""
    if val is None:
        return 0.0
    s = val.replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _is_time_metric(name: str) -> bool:
    """Heuristically detect time-related metrics for automatic unit conversion."""
    lower = name.lower()
    return "duration" in lower or "time" in lower or "elapsed" in lower


def _is_percentage_metric(name: str) -> bool:
    """Heuristically detect percentage metrics."""
    lower = name.lower()
    # derived__pct_occupancy_per_* are ratios (slope), not percentages.
    if "derived__pct_occupancy" in lower:
        return False
    # Split by common delimiters and check for exact token 'pct' to avoid
    # false positives (e.g. 'duration' does NOT contain 'pct' as a token).
    tokens = re.split(r"[._\s]", lower)
    return "pct" in tokens or "percent" in lower


def _aggregate_metric(name: str, values: list[float]) -> float:
    """Aggregate metric values across multiple kernel invocations.

    NCU suffix semantics:
      .avg -> average across invocations
      .max -> max across invocations
      .min -> min across invocations
      .sum -> sum across invocations
      (no suffix) -> if constant (launch config), keep it; else sum
    """
    if not values:
        return 0.0
    if name.endswith(".avg"):
        return sum(values) / len(values)
    elif name.endswith(".max"):
        return max(values)
    elif name.endswith(".min"):
        return min(values)
    elif name.endswith(".sum"):
        return sum(values)
    else:
        # No suffix: launch config metrics should be averaged if they vary;
        # counters like sm__cycles_active should be summed.
        if len(set(values)) == 1:
            return values[0]
        if name.lower().startswith("launch__"):
            return sum(values) / len(values)
        return sum(values)


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    """Find the first candidate column name that exists in cols (case-insensitive)."""
    col_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def _is_long_format(rows: list[dict]) -> bool:
    """Detect whether CSV is long format (one metric per row)."""
    if not rows:
        return False
    cols = [c.lower() for c in rows[0].keys()]
    return "metric name" in cols and "metric value" in cols


def parse_long_format(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Parse long-format NCU CSV: each row is one metric for one kernel invocation."""
    cols = list(rows[0].keys())
    kernel_col = _find_col(cols, ["Kernel Name", "Kernel"])
    metric_col = _find_col(cols, ["Metric Name", "Metric"])
    value_col = _find_col(cols, ["Metric Value", "Value"])
    unit_col = _find_col(cols, ["Metric Unit", "Unit"])

    if not all([kernel_col, metric_col, value_col]):
        print("Error: Could not identify required columns in long-format CSV.", file=sys.stderr)
        print(f"Columns found: {cols}", file=sys.stderr)
        sys.exit(1)

    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        kname = row.get(kernel_col, "Unknown").strip()
        if not kname:
            continue
        mname = row.get(metric_col, "").strip()
        if not mname or mname.lower().startswith("device__attribute_"):
            continue
        mval = _to_float(row.get(value_col, ""))
        raw[kname][mname].append(mval)

    kernels: dict[str, dict[str, float]] = {}
    for kname, metrics in raw.items():
        kernels[kname] = {mname: _aggregate_metric(mname, vals) for mname, vals in metrics.items()}
    return kernels


def _discover_metric_columns(cols: list[str]) -> list[str]:
    """
    Discover metric columns in wide-format CSV.
    We exclude known non-metric columns (ID, Process, Host, Kernel, Block, Grid, etc.)
    and treat everything else as a metric.
    """
    non_metric_patterns = [
        r"^id$",
        r"^process\s*id$",
        r"^process\s*name$",
        r"^host\s*name$",
        r"^kernel\s*name$",
        r"^kernel\s*time$",
        r"^context$",
        r"^stream$",
        r"^block\s*size$",
        r"^grid\s*size$",
        r"^device$",
        r"^ invocation",
        r"^device__attribute_",
    ]
    metrics = []
    for c in cols:
        c_lower = c.lower().strip()
        is_metric = True
        for pat in non_metric_patterns:
            if re.match(pat, c_lower):
                is_metric = False
                break
        if is_metric:
            metrics.append(c)
    return metrics


def parse_wide_format(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Parse wide-format NCU CSV: each row is one kernel, metrics are columns."""
    cols = list(rows[0].keys())
    kernel_col = _find_col(cols, ["Kernel Name", "Kernel"])
    if not kernel_col:
        print("Error: Could not identify Kernel Name column in wide-format CSV.", file=sys.stderr)
        print(f"Columns found: {cols}", file=sys.stderr)
        sys.exit(1)

    metric_cols = _discover_metric_columns(cols)
    if not metric_cols:
        print("Warning: No metric columns discovered in wide-format CSV.", file=sys.stderr)
        print(f"Columns found: {cols}", file=sys.stderr)

    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        kname = row.get(kernel_col, "Unknown").strip()
        if not kname:
            continue
        for mcol in metric_cols:
            mname = mcol.strip()
            if mname.lower().startswith("device__attribute_"):
                continue
            mval = _to_float(row.get(mcol, ""))
            raw[kname][mname].append(mval)

    kernels: dict[str, dict[str, float]] = {}
    for kname, metrics in raw.items():
        kernels[kname] = {mname: _aggregate_metric(mname, vals) for mname, vals in metrics.items()}
    return kernels


def parse_ncu_csv(path: str) -> dict[str, dict[str, float]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Error: CSV file is empty.", file=sys.stderr)
        sys.exit(1)

    if _is_long_format(rows):
        return parse_long_format(rows)
    else:
        return parse_wide_format(rows)


def compute_derived(kernels: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Add derived metrics (bandwidth, total bytes, etc.) to each kernel."""
    for kname, metrics in kernels.items():
        # Try to find dram read/write metrics with various naming conventions
        read_b = 0.0
        write_b = 0.0
        time_ns = 0.0

        for mname, mval in metrics.items():
            lower = mname.lower()
            # Prefer .sum variants for total bandwidth computation
            if "dram__bytes_read.sum" in lower:
                read_b = mval
            elif "dram__bytes_read" in lower and read_b == 0.0:
                read_b = mval
            if "dram__bytes_write.sum" in lower:
                write_b = mval
            elif "dram__bytes_write" in lower and write_b == 0.0:
                write_b = mval
            if "gpu__time_duration.sum" in lower:
                time_ns = mval
            elif "gpu__time_duration" in lower and time_ns == 0.0:
                time_ns = mval

        metrics["__total_dram_mb"] = (read_b + write_b) / (1024 * 1024)
        if time_ns > 0:
            metrics["__bandwidth_gbps"] = (read_b + write_b) / time_ns
            metrics["__time_us"] = time_ns * 1e-3
        else:
            metrics["__bandwidth_gbps"] = 0.0
            metrics["__time_us"] = 0.0

        # Approximate arithmetic intensity using FFMA if available
        ffma = 0.0
        for mname, mval in metrics.items():
            if "ffma" in mname.lower() and "executed" in mname.lower():
                ffma = mval
                break
        total_bytes = read_b + write_b
        if total_bytes > 0:
            metrics["__arith_intensity"] = (ffma * 2) / total_bytes
        else:
            metrics["__arith_intensity"] = 0.0

    return kernels


def _fmt_val(name: str, val: float) -> str:
    """Format a metric value with appropriate units."""
    if _is_percentage_metric(name):
        return f"{val:.2f}%"
    if "byte" in name.lower() and val > 1024 * 1024:
        return f"{val / (1024 * 1024):.2f} MB"
    if _is_time_metric(name) and val > 1000:
        return f"{val * 1e-3:.2f} us"
    if abs(val) >= 1e6:
        return f"{val:.3e}"
    if abs(val) >= 1.0:
        return f"{val:.2f}"
    return f"{val:.4f}"


def print_report(kernels: dict[str, dict[str, float]], sort_by: str = "time"):
    """Print a human-readable report."""
    sort_key_map = {
        "time": lambda kv: kv[1].get("__time_us", 0.0),
        "name": lambda kv: kv[0],
        "sm": lambda kv: next(
            (v for k, v in kv[1].items() if "throughput" in k.lower() and "pct" in k.lower()), 0.0
        ),
    }
    sort_key = sort_key_map.get(sort_by, sort_key_map["time"])
    sorted_kernels = sorted(kernels.items(), key=sort_key, reverse=(sort_by != "name"))

    total_time = sum(m.get("__time_us", 0.0) for m in kernels.values())

    # Build the summary table header
    print("=" * 120)
    print(
        f"{'Kernel Name':<50} {'Time(us)':>10} {'%Total':>8} "
        f"{'SM%':>8} {'Reg':>6} {'DRAM(MB)':>10} {'BW(GB/s)':>10} {'ArithInt':>10}"
    )
    print("-" * 120)

    for kname, metrics in sorted_kernels:
        time_us = metrics.get("__time_us", 0.0)
        pct_total = (time_us / total_time * 100) if total_time > 0 else 0.0

        # Find SM utilization metric (various possible names)
        sm_util = 0.0
        for k, v in metrics.items():
            if "throughput" in k.lower() and "pct" in k.lower():
                sm_util = v
                break

        # Find register metric
        reg = 0.0
        for k, v in metrics.items():
            if "register" in k.lower() and "per_thread" in k.lower():
                reg = v
                break

        dram_mb = metrics.get("__total_dram_mb", 0.0)
        bw = metrics.get("__bandwidth_gbps", 0.0)
        arith = metrics.get("__arith_intensity", 0.0)

        print(
            f"{kname:<50} {time_us:>10.2f} {pct_total:>7.1f}% "
            f"{sm_util:>7.1f}% {reg:>6.0f} {dram_mb:>10.2f} {bw:>10.2f} {arith:>10.3f}"
        )

    print("-" * 120)
    print(f"{'Total':<50} {total_time:>10.2f} {'100.0%':>8}")
    print("=" * 120)

    # Print ALL metrics for each kernel (full disclosure)
    print("\n" + "=" * 120)
    print("FULL METRIC BREAKDOWN (all metrics found in CSV)")
    print("=" * 120)

    for kname, metrics in sorted_kernels:
        print(f"\n{'─' * 120}")
        print(f"Kernel: {kname}")
        print(f"{'─' * 120}")

        # Separate derived metrics from raw metrics
        raw_metrics = {k: v for k, v in metrics.items() if not k.startswith("__")}
        derived_metrics = {k: v for k, v in metrics.items() if k.startswith("__")}

        # Sort raw metrics alphabetically for stable output
        for mname in sorted(raw_metrics.keys()):
            mval = raw_metrics[mname]
            display = _fmt_val(mname, mval)
            print(f"  {mname:<60} {display:>20}")

        if derived_metrics:
            print(f"  {'─' * 80}")
            for mname in sorted(derived_metrics.keys()):
                mval = derived_metrics[mname]
                display = _fmt_val(mname, mval)
                print(f"  {mname:<60} {display:>20}")


def export_json(kernels: dict, path: str):
    with open(path, "w") as f:
        json.dump(kernels, f, indent=2)
    print(f"\nReport exported to {path}")


def main():
    parser = argparse.ArgumentParser(description="Parse NCU CSV and generate report")
    parser.add_argument("csv_file", help="Path to NCU CSV output")
    parser.add_argument("--kernel", help="Filter kernels by name substring (case-insensitive)")
    parser.add_argument("--sort", choices=["time", "name", "sm"], default="time", help="Sort order")
    parser.add_argument("--json", metavar="PATH", help="Export full report to JSON")
    args = parser.parse_args()

    kernels = parse_ncu_csv(args.csv_file)
    kernels = compute_derived(kernels)

    if args.kernel:
        kernels = {
            k: v for k, v in kernels.items() if args.kernel.lower() in k.lower()
        }
        if not kernels:
            print(f"No kernels matched filter: {args.kernel}", file=sys.stderr)
            sys.exit(1)

    print_report(kernels, sort_by=args.sort)

    if args.json:
        export_json(kernels, args.json)


if __name__ == "__main__":
    main()
