<div align="center">

# ⚡ Mini-vLLM

**A light, transparent, and modular inference & quantization engine for studying LLMs.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

</div>

## 🌟 Features

- **Transparent Architecture**: Cleanly separated modules for Model, Engine, and Kernels.
- **AWQ Quantization**: Built-in 4-bit AWQ quantization support with calibration.
- **KV Cache Management**: Simple Key-Value cache management for autoregressive decoding.
- **W^T Layout & Fused Kernels**: Triton kernels with pre-transposed weight layout and fused dequantization for lower memory traffic.
- **CUDA Graph Acceleration**: Pre-captured CUDA Graphs for the decode phase to eliminate CPU launch overhead.
- **Fused CUDA Megakernel**: Optional persistent megakernel backend that fuses embedding + all transformer layers + final norm + LM head into a single kernel launch.
- **Profiler Trace Analysis**: Built-in script to parse PyTorch profiler traces and diagnose kernel-level bottlenecks.


---

## 🚀 Quick Start

### 1. Environment Setup

We highly recommend using [`uv`](https://github.com/astral-sh/uv) for ultra-fast dependency installation.

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

Download your model (e.g., Qwen3-0.6B) to your local path.

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-1.7B/
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

**Megakernel backend** — edit `configs/default.yaml`:
```yaml
inference:
  backend: "megakernel_cuda"   # default | megakernel_cuda
```
Then run the same command:
```bash
python main.py
```

**Quantized model inference** — enable quantized model and set the path in config:
```yaml
inference:
  use_quantized_model: true
path:
  quantized_model_path: "~/huggingface/Qwen3-0.6B-AWQ_Cached"
```
Then run:
```bash
python main.py
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

> 📖 **Full script documentation** → see [`scripts/README.md`](scripts/README.md)

---

## ⚙️ Configuration

All behaviors are controlled by `configs/default.yaml`.

### Supported Parameters

**Inference**
```yaml
inference:
  backend: default                    # [default, megakernel_cuda]
  use_cuda_graph: true                # [true, false]
  use_kvcache: true                   # [true, false]
  use_sdpa: true                      # [true, false]
  check_correction: false             # [true, false]
  use_profile: false                  # [true, false]
  use_quantized_model: false            # [true, false]
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
├── configs/               # YAML configuration files
│   ├── default.yaml
│   └── profile.yaml
├── engine/                # Inference engine & runtime
│   ├── loader.py          # Top-level model loader
│   ├── model_runner.py    # Inference loop (prefill + decode)
│   ├── eval_runner.py     # Baseline / megakernel eval runners
│   ├── sampler.py
│   ├── context.py
│   └── progress.py
├── kernels/               # Custom kernels (Triton & CUDA)
│   ├── awq_gemm.py
│   ├── awq_gemm_wt.py
│   ├── awq_gemm_wt_fused.py
│   └── megakernel_cuda/   # Fused CUDA persistent megakernel
│       ├── decode_ldg.cu
│       ├── decode_wrapper.cpp
│       └── sm_profiler.h
├── layers/                # Built-in dense layers (Attention, MLP, RMSNorm, ...)
├── quantization/          # AWQ search, quantized linear layers, checkpoint I/O
│   ├── awq.py
│   ├── checkpoint.py
│   ├── quantized_linear.py
│   ├── quantized_linear_wt.py
│   ├── quant_math.py
│   ├── scale.py
│   ├── module_ops.py
│   └── calibration.py
├── model/                 # Model architectures
│   ├── qwen3.py
│   ├── qwen3_megakernel.py
│   └── megakernel_weights.py
├── scripts/               # Evaluation & development scripts
│   ├── bench_megakernel.py
│   ├── verify_megakernel.py
│   ├── analyze_trace.py
│   ├── parse_ncu_csv.py
│   ├── run_ncu_profile.sh
│   ├── ablate.py
│   └── README.md          # Script usage documentation
├── tools/                 # Devops / repository utilities
│   └── bundle_sync.py
├── utils/                 # Cross-cutting helpers (Config, Logger, CPU offload, ...)
│   ├── config.py
│   ├── logger.py
│   ├── model_loader.py
│   ├── cpu_offload.py
│   ├── bench_harness.py
│   └── verifier.py
├── main.py                # Entry point: inference (fp16 / bf16 / quantized / megakernel)
└── quant.py               # Entry point: AWQ quantization calibration
```

**Entry points:**
- `main.py` — unified inference (auto-detects quantized weights, supports `backend: megakernel_cuda`)
- `quant.py` — AWQ quantization calibration

**No separate `run_quantized.py`** — quantized inference is handled by `main.py`.

---

## 🗺️ Roadmap

- [x] **Autoregressive Decoding & KV Cache**: Basic generation loop with Key-Value cache management.
- [x] **AWQ Quantization & Forward Pass**: 4-bit calibration and quantized linear layer forward implementation.
- [x] **Quantized Kernel Support**: Integrate CUDA/Triton kernels for 4-bit model inference.
- [x] **Quantized Model Persistence**: Support saving and loading calibrated AWQ weights.
- [x] **W^T Layout & Fused Triton Kernels**: Pre-transposed weight layout with fused kernels.
- [x] **CUDA Graph Decode Acceleration**: Pre-capture decode graphs to remove CPU launch overhead.
- [x] **Profiler Trace Analyzer**: Parse PyTorch profiler JSON and report kernel time breakdown.
- [x] **Fused CUDA Megakernel**: Persistent megakernel backend with single-kernel decode pipeline.
- [ ] **Implement EOS**: Ensure generation loops break correctly on EOS tokens.

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
