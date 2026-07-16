<div align="center">

# Mini-vLLM
A lightweight inference and quantization engine for studying LLMs.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
</div>

## Decode Throughput 

> Measured on RTX 4050 6 GB · greedy · 128 decode tokens

| Mode | Model | HF Transformers | mini-vllm | Speedup |
|------|-------|----------------|-----------|---------|
| Single-request | Qwen3-0.6B | 23.0 tok/s | 114.5 tok/s (megakernel) | **5.0x** |
| Single-request | Qwen3.5-0.8B | 15.3 tok/s | 90.9 tok/s (CUDA Graph + flash-attn + fla) | **6.0x** |
| Batch (bs=8) | Qwen3-0.6B | — | 110 tok/s (paged KV + CUDA Graph) | — |

## Streaming Inference & Megakernel 

<div align="center">
<img src="assets/gifs/perf_compare.gif" alt="mini-vllm megakernel vs HF Transformers — same Qwen3-0.6B, same prompt, ~8x faster" />

<sub>Qwen3-0.6B · bf16 · greedy · same prompt · 6 GB RTX 4050<br/>Left: HF Transformers · Right: mini-vllm megakernel</sub>

</div>

## Features

- **HTTP Serving** -- OpenAI-compatible `/v1/chat/completions` with streaming and concurrent request batching
- **Continuous Batching** -- dynamic scheduler with FIFO / SPF / LJF / random admission and chunked prefill
- **Paged KV Cache** -- FA2 paged attention (`block_table`-based), eliminates full-sequence KV gather at decode
- **Multi-batch CUDA Graph** -- bucketed graph capture across batch sizes [1,2,4,8,...]; decode replay with static paged buffers
- **Flash Attention** -- varlen FA2 prefill, `flash_attn_with_kvcache` paged decode, SDPA fallback
- **Fused CUDA Megakernel** -- single-kernel decode fusing all transformer layers (bs=1)
- **KV Cache Quantization (KIVI)** -- 2/4-bit asymmetric quantization with full-precision residual window
- **AWQ 4-bit Quantization** -- weight-only quantization with Triton/CUDA kernels
- **SwanLab Monitoring** -- throughput and memory tracking ([docs](docs/profiling.md))
- **lm-eval Benchmark** -- evaluation harness adapter ([docs](docs/benchmark.md))

## Supported Models

| Model | Model Backend | Attention Backend |
|-------|--------------|-------------------|
| Qwen3 | `default`, `megakernel_cuda` | `sdpa`, `flash_attn`, `naive` |
| Qwen3.5 | `default` | `sdpa`, `flash_attn`, `naive`, `fla` |

## Quick Start

### 1. Install

```bash
git clone https://github.com/BoundlessWindMoon/minivllm.git
cd mini-vllm

uv venv .venv --python 3.12
source .venv/bin/activate

# PyTorch (match your CUDA version)
uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
uv pip install -e .
```

**Optional extras:**

```bash
# Qwen3.5: flash-attn and flash-linear-attention are required
pip install flash-attn --no-build-isolation
pip install flash-linear-attention --no-build-isolation

uv pip install -e ".[benchmark]"   # lm-eval
uv pip install -e ".[monitor]"     # SwanLab
uv pip install -e ".[quant]"       # AWQ
```

### 2. Download Model

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/

# Qwen3.5 requires flash-attn + flash-linear-attention
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3.5-0.8B \
  --local-dir ~/huggingface/Qwen3.5-0.8B/
```

### 3. Run

```bash
# Qwen3, megakernel backend
python main.py

# Qwen3.5, CUDA Graph (requires flash-attn + flash-linear-attention)
python main.py --config configs/qwen3_5.yaml

# Continuous batching
python batch_main.py
python batch_main.py --sweep-policies fifo spf ljf random
python batch_main.py --repeat 5

# HTTP server (OpenAI-compatible)
python scripts/serve/start_server.py --config configs/runs/batch.yaml
python scripts/serve/bench_serving.py --mode sweep   # concurrent load test
```

### 4. Benchmark

```bash
python -m eval.run --config configs/qwen3_5.yaml --tasks arc_easy --limit 50
```

See [docs/benchmark.md](docs/benchmark.md) for task customization and log output.

## Documentation

- [Configuration Reference](docs/config.md) -- all YAML parameters
- [Benchmark Evaluation](docs/benchmark.md) -- lm-eval integration
- [Profiling & Monitoring](docs/profiling.md) -- PyTorch profiler, Perfetto, SwanLab
- [AWQ Quantization](docs/quantization.md) -- calibration and quantized inference

## Project Structure

```text
mini-vllm/
├── configs/            # YAML configs
├── engine/             # Inference loop, model runner, scheduler, batched runner, KV pool
├── kernels/            # Triton & CUDA kernels (AWQ, KIVI)
├── layers/             # Attention, MLP, RMSNorm, Rotary, KV cache backends
├── model/              # Qwen3 / Qwen3.5 architectures
├── quantization/       # AWQ calibration and quantized layers
├── eval/               # lm-eval adapter
├── scripts/            # Benchmark, verify, profile tools
├── test/               # Functional test suite (pytest)
├── assets/prompts/     # JSONL prompt workloads for batch testing
├── main.py             # Entry point: single-request inference
├── batch_main.py       # Entry point: continuous batching
└── quant_cli.py        # Entry point: AWQ calibration
```

## Acknowledgements

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) -- minimalist inference architecture
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) -- AWQ quantization
- [mega-qwen](https://github.com/coffee0224/mega-qwen) -- fused megakernel design

## License

MIT
