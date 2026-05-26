<div align="center">

# ⚡ Mini-vLLM

**A light, transparent, and modular inference & quantization engine for studying LLMs.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

<br/>

<img src="docs/perf_compare.gif" alt="mini-vllm megakernel vs HF Transformers — same Qwen3-0.6B, same prompt, ~8× faster" />

<sub>Qwen3-0.6B · bf16 · greedy · same prompt · 6 GB RTX 4050<br/>Left: HF Transformers · Right: mini-vllm megakernel</sub>

</div>

## 🌟 Features

- **Fused CUDA Megakernel**: Single-kernel decode pipeline that fuses embedding, all transformer layers, final norm and LM head — ~8× faster than HF Transformers on qwen3-0.6B.
- **AWQ Quantization**: Built-in 4-bit calibration and inference with Triton/CUDA kernels and pre-transposed W^T layout.
- **CUDA Graph Decode**: Configurable bucketed CUDA Graphs to eliminate CPU launch overhead during autoregressive generation.
- **Profiler & Trace Analysis**: PyTorch profiler integration with Perfetto UI support and automated kernel-level bottleneck reports.


---

## 🚀 Quick Start

### 1. Environment Setup

using [`uv`](https://github.com/astral-sh/uv) for dependency installation.

```bash
# Clone the repository
git clone https://github.com/BoundlessWindMoon/minivllm.git
cd mini-vllm

# Create and activate virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install mini-vllm and all dependencies
uv pip install -e ".[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### 2. Prepare Model

Download a supported model (e.g., **Qwen3-0.6B** or **Qwen3.5-0.8B**) to your local path.

```bash
# Qwen3-0.6B (default config)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/

# Qwen3.5-0.8B (multimodal + linear attention)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ~/huggingface/Qwen3.5-0.8B/
```

Update the path in `configs/default.yaml`:
```yaml
path:
  model_path: "~/huggingface/Qwen3-0.6B/"
```

### 3. Run Inference

**Standard inference** (PyTorch eager / CUDA Graph):
```bash
python main.py
```

Or use your own config file:
```bash
python main.py --config configs/qwen3_5.yaml
```

**Megakernel backend** — edit `configs/default.yaml`:
```yaml
inference:
  backend: "megakernel_cuda"   # default | megakernel_cuda
```


**Quantized model inference** — enable quantized model and set the path in config:
```yaml
inference:
  use_quantized_model: true
path:
  quantized_model_path: "~/huggingface/Qwen3-0.6B-AWQ_Cached"
```

### 4. Run Quantization (AWQ)

Run the quantization pipeline and save calibrated weights:
```bash
python quant.py
```

### 5. Benchmark & Verify

```bash
# Verify megakernel correctness against baseline
python scripts/verify_megakernel.py --steps 10

# Benchmark decode throughput (baseline vs megakernel)
python scripts/bench_megakernel.py --backend both --input-len 32 --output-len 128

# Profile a quantized model with NCU
bash ./scripts/run_ncu_profile.sh --full

# Analyze PyTorch profiler trace
python scripts/analyze_trace.py log/profile/*.pt.trace.json
```

### Profiling with PyTorch Profiler + Perfetto

Enable profiling in `configs/default.yaml`:
```yaml
inference:
  use_profile: true
path:
  profile_dir: ./log/profile/
```

After inference, open the latest trace in **Perfetto UI** (default):

```bash
# Auto-detect latest trace and open in browser
python scripts/open_profile.py

# Open a specific trace file
python scripts/open_profile.py log/profile/some_trace.pt.trace.json

```
---

## ⚙️ Configuration

All behaviors are controlled by `configs/default.yaml`.

### Supported Parameters

**Inference**
```yaml
inference:
  backend: default                    # [default, megakernel_cuda]
  megakernel_variant: default         # [default, naive, p0-p10, all_combined]
  use_cuda_graph: true                # [true, false]
  use_kvcache: true                   # [true, false]
  use_sdpa: true                      # [true, false]
  check_correction: false             # [true, false]
  use_profile: false                  # [true, false]
  use_quantized_model: false          # [true, false]
  stop_on_eos: true                   # [true, false]
  use_chat_template: false            # [true, false]
  use_thinking: true                  # [true, false] (only when use_chat_template=true)
  cpu_offload_modules: []             # submodules to keep on CPU, e.g. [model.embed_tokens]
  max_new_tokens: 128                 # integer
  prompt: "Hello, I am ..."           # string
  sampling:
    sample_method: greedy             # [greedy, topp]
    temperature: 1.0                  # float
    topk: 1                           # integer
    topp: 1.0                         # float (0.0 ~ 1.0)
```

**Environment**
```yaml
env:
  device: cuda:0                      # string, e.g. cuda:0, cuda:1
  default_dtype: bfloat16             # [float16, bfloat16]
  distributed:
    backend: nccl                     # [nccl, gloo]
    world_size: 1                     # integer
    rank: 0                           # integer
    init_method: tcp://localhost:29500  # string
```

**Path**
```yaml
path:
  model_path: ~/huggingface/Qwen3-0.6B/   # string
  baseline_model_path: ~/huggingface/baseline/  # string
  data_path: ''                           # string (optional)
  quantized_model_path: ''                # string (optional)
  profile_dir: ./log/profile/             # string
```

**Quantization**
```yaml
quant:
  quant_method: AWQ                   # [AWQ]
  quant_bits: 4                       # [4]
  quant_targets: [MLP, ATTENTION]     # [MLP, ATTENTION, LM_HEAD]
  group_size: 128                     # [128]
  has_zero_point: true                # [true, false]
  apply_clip: true                    # [true, false]
  export_compatible: false            # [true, false]
  backend: gemm                       # [gemm, triton, triton_wt, triton_wt_fused]
  layout: Wt                          # [Wt]
  pack_order: sequential              # [sequential]
  max_chunk_memory: 1073741824        # integer (bytes)
  calibration:
    data: pileval                     # string
    n_samples: 32                     # integer
    max_seq_length: 512               # integer
    split: train                      # string
    text_column: text                 # string
```

---

## 📂 Project Structure

```text
mini-vllm/
├── configs/               # YAML configs (default.yaml, qwen3_5.yaml, ...)
├── engine/                # Inference loop, model runner, eval runners, sampler
├── kernels/               # Triton & CUDA kernels (AWQ gemm, fused megakernel)
├── layers/                # Attention, MLP, RMSNorm, Rotary, Gated Delta Rule
├── model/                 # Qwen3 / Qwen3.5 architectures, megakernel variants
├── quantization/          # AWQ calibration, quantized layers, checkpoint I/O
├── scripts/               # Benchmark, verify, profile, analysis tools
├── tools/                 # Devops utilities
├── utils/                 # Config, logger, model loader, verifier
├── main.py                # Entry point: inference
└── quant.py               # Entry point: AWQ calibration
```

- `main.py --config <path>` to use a custom config.
- Quantized inference is handled by `main.py`; no separate runner.

---

## 🗺️ Roadmap

- [x] Autoregressive decoding with KV cache, CUDA Graph acceleration, and EOS handling
- [x] AWQ 4-bit quantization with calibration, W^T layout, and Triton/CUDA kernels
- [x] Fused CUDA megakernel (single-kernel decode pipeline)
- [x] PyTorch profiler integration with trace analysis

---

## 🙏 Acknowledgements

This project is inspired by and references the following excellent open-source works or utils:

- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**: For the minimalist and transparent LLM inference architecture design.
- **[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)**: For the robust and efficient AWQ quantization algorithm implementation.
- **[GLM](https://glm-5.org/zh/)**: For the intelligent coding assistance and infrastructure debugging support.
- **[mega-qwen](https://github.com/coffee0224/mega-qwen)**: A high-performance inference engine for Qwen3-0.6B built around a fused CUDA megakernel architecture.

---

## 📜 License

This project is licensed under the MIT License.
