# Scripts

## Serving (`scripts/serve/`)

| Script | Purpose |
|--------|---------|
| `start_server.py` | Start the HTTP server. See [docs/features/serving.md](../docs/features/serving.md). |
| `bench_serving.py` | Benchmark throughput and latency (fixed / poisson / sweep modes). |
| `chat_client.py` | Interactive multi-turn chat client over HTTP. |

## Evaluation (`scripts/eval/`)

| Script | Purpose |
|--------|---------|
| `verify_e2e.py` | Compare mini-vllm greedy output against HuggingFace baseline token-by-token. |
| `verify_kivi.py` | Check KIVI quantized KV output matches the dense baseline within tolerance. |
| `download_eval_datasets.sh` | Download lm-eval datasets for offline evaluation. |
| `sweep_overnight.sh` | Run lm-eval across multiple tasks unattended. |

## Profiling (`scripts/profile/`)

| Script | Purpose |
|--------|---------|
| `view_trace.py` | Serve a PyTorch profiler trace in Perfetto UI (opens browser). |
| `analyze_trace.py` | Parse a Chrome trace JSON and summarize CUDA kernel time by category. |
| `upload_profile.py` | Upload a trace to a remote viewer. |

## Tools (`scripts/tools/`)

| Script | Purpose |
|--------|---------|
| `check_docs.py` | Verify `code-ref` annotations in docs point to existing source files. Run with `make check-docs`. |
| `convert_hf_awq.py` | Convert a HuggingFace AWQ checkpoint to mini-vllm format. |
| `download_sharegpt.py` | Download the ShareGPT dataset for serving benchmarks. |
| `download_workloads.py` | Download prompt workload JSONL files. |
| `bundle_sync.py` | Pack/unpack the repo as a git bundle for offline transfer. |

## Ablation (`scripts/ablation/`)

Kernel-level micro-benchmarks for attention and matmul variants.
Intended for development; not required for normal use.
