# Scripts & Tools

This directory contains standalone evaluation scripts, profiling tools, and utility scripts for mini-vllm.

All Python scripts share a common pattern: they read `configs/default.yaml` (or a custom config via `--config`) for model paths, device settings, and inference parameters. This avoids hardcoding paths in individual scripts.

---

## Benchmark & Verification

### `bench_megakernel.py`

Benchmark decode throughput: baseline (PyTorch eager / CUDA Graph) vs fused CUDA megakernel.

```bash
# Compare both backends (default)
python scripts/bench_megakernel.py --backend both --input-len 32 --output-len 128

# Baseline only, with CUDA Graph
python scripts/bench_megakernel.py --backend baseline --num-warmup 20 --num-runs 50

# Baseline without CUDA Graph (for ablation)
python scripts/bench_megakernel.py --backend baseline --no-cuda-graph

# Megakernel only
python scripts/bench_megakernel.py --backend megakernel --input-len 128 --output-len 256

# Use a custom config
python scripts/bench_megakernel.py --config configs/my_config.yaml --backend both
```

**Key options:**
- `--backend {baseline,megakernel,both}` — which backend to benchmark
- `--input-len` — prefill prompt length (random tokens)
- `--output-len` — number of decode steps to measure
- `--num-warmup` — warmup iterations before timed runs
- `--num-runs` — number of timed runs for statistics
- `--no-cuda-graph` — disable CUDA Graph for baseline (shows raw PyTorch eager performance)

---

### `verify_megakernel.py`

Correctness verification: run greedy decode on both baseline and megakernel with the same prompt, compare tokens and logits step-by-step.

```bash
# Default: 10 steps, prompt "The capital of France is"
python scripts/verify_megakernel.py

# Custom step count and prompt
python scripts/verify_megakernel.py --steps 20 --prompt "Once upon a time"

# Use custom config
python scripts/verify_megakernel.py --config configs/default.yaml --steps 5
```

**Pass criteria:**
- All tokens match
- Max logits diff < 0.5
- Cosine similarity > 0.999

---

## Profiling & Analysis

### `run_ncu_profile.sh`

Run Nsight Compute (NCU) profiling on the quantized model and generate a parsed report.

```bash
# Standard mode (raw page metrics)
bash scripts/run_ncu_profile.sh

# Full metrics (slower, ~5-10x overhead)
bash scripts/run_ncu_profile.sh --full

# Custom output directory
bash scripts/run_ncu_profile.sh --full ./log/ncu_custom/
```

**Prerequisites:**
- `ncu` CLI must be installed and GPU performance counters accessible.
- If you see `ERR_NVGPUCTRPERM`, run:
  ```bash
  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
  ```

**Output:**
- `log/ncu/profile.ncu-rep` — binary NCU report
- `log/ncu/profile.csv` — CSV export
- `log/ncu/profile_report.txt` — human-readable parsed report

---

### `parse_ncu_csv.py`

Parse NCU CSV output and print a performance report. Can be used standalone or is automatically invoked by `run_ncu_profile.sh`.

```bash
# Basic report
python scripts/parse_ncu_csv.py log/ncu/profile.csv

# Filter by kernel name
python scripts/parse_ncu_csv.py log/ncu/profile.csv --kernel "awq_gemm" --sort time

# Export to JSON
python scripts/parse_ncu_csv.py log/ncu/profile.csv --json report.json
```

---

### `analyze_trace.py`

Parse PyTorch profiler Chrome trace JSON and categorize CUDA kernel time.

```bash
# Analyze a single trace
python scripts/analyze_trace.py log/profile/*.pt.trace.json

# Compare two traces side-by-side
python scripts/analyze_trace.py --compare \
    log/profile/quant_trace.pt.trace.json \
    log/profile/fp16_trace.pt.trace.json
```

**Categories:** AWQ Linear, cuBLAS Matmul, RMSNorm, SDPA / Attention, Activation, Embedding / LM Head, KV Cache, Memory / Copy, Other.

---

## Development Utilities

### `tools/bundle_sync.py`

Pack or unpack the mini-vllm repository using git bundle (useful for offline transfer).

```bash
# Pack current branch into a bundle
python tools/bundle_sync.py pack mini-vllm.bundle

# Unpack on target machine
python tools/bundle_sync.py unpack mini-vllm.bundle ./mini-vllm

# Restore origin remote in a bundle-cloned repo
python tools/bundle_sync.py setup-remote
```

---

### `ablate.py`

Triton kernel ablation study: fixed-config micro-benchmarks for AWQ / FP16 matmul kernels. Used for kernel development and tuning, not for end-to-end model evaluation.

```bash
# Run all kernel variants with fixed configs (no autotune)
python scripts/ablate.py
```

**Note:** This script is an internal development tool. It directly calls Triton kernels without going through the model / engine layers.

---

## Common Configuration

All benchmark / verification scripts respect `configs/default.yaml`. The most relevant fields are:

```yaml
path:
  model_path: ~/huggingface/Qwen3-0.6B/       # Base model path
  data_path: ''                               # Override data path (optional)
  quantized_model_path: ''                    # Quantized model path (optional)

env:
  device: cuda:0
  default_dtype: float16                      # or bfloat16

inference:
  backend: default                            # default | megakernel_cuda
  use_quantized_model: false                    # true to load quantized model
  use_cuda_graph: true
  use_sdpa: true
```

To switch to the megakernel backend for `main.py`, set:
```yaml
inference:
  backend: "megakernel_cuda"
```
