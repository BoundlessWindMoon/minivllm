"""Parse PyTorch profiler Chrome trace JSON and categorize CUDA kernel time.

Usage:
    # Analyze a single trace
    python scripts/analyze_trace.py log/profile/*.pt.trace.json

    # Compare quantized vs fp16 trace side-by-side
    python scripts/analyze_trace.py --compare \
        log/profile/quant_trace.pt.trace.json \
        log/profile/fp16_trace.pt.trace.json

Categorizes kernels into:
  - AWQ Linear (quantized matmul)
  - cuBLAS Matmul (fp16 baseline)
  - RMSNorm / LayerNorm
  - SDPA / Attention
  - Activation / Elementwise
  - Embedding / LM Head
  - Memory / Copy
  - Other
"""
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def categorize_kernel(name: str) -> str:
    """Map a CUDA kernel name to a functional category."""
    n = name.lower()

    if "awq_gemm" in n or ("awq" in n and "gemm" in n):
        return "AWQ Linear"

    if "cublas" in n or "cutlass" in n:
        return "cuBLAS Matmul"

    if "rms_norm" in n or "layernorm" in n:
        return "RMSNorm"

    if any(k in n for k in ("sdpa", "flash", "cudnn::", "attention")):
        return "SDPA / Attention"

    if "rotary" in n:
        return "Rotary Embedding"

    if any(k in n for k in ("silu", "gelu", "relu", "elementwise", "vectorized")):
        return "Activation / Elementwise"

    if "embedding" in n:
        return "Embedding / LM Head"

    if "kvcache" in n or "cache" in n:
        return "KV Cache"

    if "memcpy" in n or "copy" in n or "fill" in n:
        return "Memory / Copy"

    return "Other"


def collect_kernel_events(trace_path: Path):
    """Load trace and return list of (name, dur_us) for kernel events."""
    with open(trace_path, "r") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    kernels = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "kernel":
            continue
        dur = ev.get("dur", 0)
        if dur <= 0:
            continue
        name = ev.get("name", "unknown")
        kernels.append((name, dur))
    return kernels


def analyze_kernels(kernels):
    """Aggregate kernel durations by name and category."""
    kernel_dur = defaultdict(float)
    category_dur = defaultdict(float)
    total = 0.0

    for name, dur in kernels:
        kernel_dur[name] += dur
        cat = categorize_kernel(name)
        category_dur[cat] += dur
        total += dur

    return kernel_dur, category_dur, total


def print_report(title: str, kernel_dur, category_dur, total):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  Total kernel time: {total/1000:.2f} ms")
    print(f"{'-'*80}")

    print(f"\n  {'Category':<30s} {'Time (ms)':>12s} {'Pct':>8s}")
    print(f"  {'-'*52}")
    for cat, dur in sorted(category_dur.items(), key=lambda x: -x[1]):
        pct = dur / total * 100
        print(f"  {cat:<30s} {dur/1000:>12.2f} {pct:>7.1f}%")
    print(f"  {'-'*52}")
    print(f"  {'TOTAL':<30s} {total/1000:>12.2f} {100.0:>7.1f}%")

    print(f"\n  {'Top 15 Individual Kernels':^78}")
    print(f"  {'Kernel Name':<48s} {'Time(ms)':>10s} {'Pct':>7s} {'Category':<16s}")
    print(f"  {'-'*84}")
    for name, dur in sorted(kernel_dur.items(), key=lambda x: -x[1])[:15]:
        pct = dur / total * 100
        cat = categorize_kernel(name)
        # truncate long names
        display_name = name[:47] if len(name) <= 47 else name[:44] + "..."
        print(f"  {display_name:<48s} {dur/1000:>10.2f} {pct:>6.1f}% {cat:<16s}")

    # Quantized vs Non-quantized
    quant_time = category_dur.get("AWQ Linear", 0.0)
    cublas_time = category_dur.get("cuBLAS Matmul", 0.0)
    non_quant_time = total - quant_time

    print(f"\n  {'Quantized vs Non-Quantized':^78}")
    print(f"  {'-'*52}")
    print(f"  {'Quantized (AWQ Linear)':<30s} {quant_time/1000:>12.2f} {quant_time/total*100:>7.1f}%")
    print(f"  {'cuBLAS Matmul (fp16)':<30s} {cublas_time/1000:>12.2f} {cublas_time/total*100:>7.1f}%")
    print(f"  {'Non-Quantized (rest)':<30s} {non_quant_time/1000:>12.2f} {non_quant_time/total*100:>7.1f}%")
    print(f"  {'-'*52}")

    if quant_time > 0:
        projected_total_2x = total - quant_time * 0.5
        speedup_2x = total / projected_total_2x
        print(f"  If AWQ Linear were 2x faster: {speedup_2x:.2f}x total speedup")
        print(f"  If AWQ + cuBLAS both 2x faster: {total / (total - quant_time*0.5 - cublas_time*0.5):.2f}x")
    print(f"{'='*80}")


def compare_reports(kernels_a, kernels_b, label_a="Quantized", label_b="FP16"):
    kd_a, cd_a, tot_a = analyze_kernels(kernels_a)
    kd_b, cd_b, tot_b = analyze_kernels(kernels_b)

    print(f"\n{'='*100}")
    print(f"  Side-by-Side Comparison: {label_a} vs {label_b}")
    print(f"{'='*100}")

    all_cats = sorted(set(cd_a.keys()) | set(cd_b.keys()),
                      key=lambda c: -(cd_a.get(c, 0) + cd_b.get(c, 0)))

    print(f"\n  {'Category':<28s} {label_a+' (ms)':>14s} {label_a+' %':>8s}   {label_b+' (ms)':>14s} {label_b+' %':>8s}   {'Delta':>10s}")
    print(f"  {'-'*90}")
    for cat in all_cats:
        da = cd_a.get(cat, 0)
        db = cd_b.get(cat, 0)
        pa = da / tot_a * 100 if tot_a > 0 else 0
        pb = db / tot_b * 100 if tot_b > 0 else 0
        delta = (da - db) / 1000
        print(f"  {cat:<28s} {da/1000:>14.2f} {pa:>7.1f}%   {db/1000:>14.2f} {pb:>7.1f}%   {delta:>+9.2f}")
    print(f"  {'-'*90}")
    print(f"  {'TOTAL':<28s} {tot_a/1000:>14.2f} {100.0:>7.1f}%   {tot_b/1000:>14.2f} {100.0:>7.1f}%   {(tot_a-tot_b)/1000:>+9.2f}")
    print(f"{'='*100}")

    ratio = tot_b / tot_a if tot_a > 0 else 0
    print(f"\n  Overall speedup: {ratio:.2f}x ({label_a} is {'faster' if ratio > 1 else 'slower'})")


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch profiler trace JSON")
    parser.add_argument("trace", nargs="+", help="Path to trace JSON file(s)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare two traces side-by-side (requires exactly 2 files)")
    args = parser.parse_args()

    paths = [Path(p) for p in args.trace]
    for p in paths:
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)

    if args.compare:
        if len(paths) != 2:
            print("--compare requires exactly 2 trace files")
            sys.exit(1)
        kernels_a = collect_kernel_events(paths[0])
        kernels_b = collect_kernel_events(paths[1])
        compare_reports(kernels_a, kernels_b, label_a=paths[0].stem, label_b=paths[1].stem)
    else:
        for p in paths:
            kernels = collect_kernel_events(p)
            kd, cd, total = analyze_kernels(kernels)
            print_report(p.name, kd, cd, total)


if __name__ == "__main__":
    main()
