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
Update the path in `configs/default.yaml`:
```yaml
path:
  model_path: "~/huggingface/Qwen3-0.6B/"
```

### 3. Run Inference
Run the standard inference pipeline using your YAML config:
```bash
python main.py
```

### 4. Run Quantization (AWQ)
Run the quantization pipeline and save it!:
```bash
python quant.py
```

### 5. Run Quantization Inference(AWQ)
Test the quantized model:
```bash
python run_quantized.py
```

## ⚙️ Configuration
You can easily control all behaviors by editing `configs/default.yaml`.

Example: Switch to Top-P Sampling
```yaml
inference:
  max_new_tokens: 256
  sampling:
    sample_method: "topp"
    temperature: 0.8
    topp: 0.95
Example: Change AWQ Calibration Settings
```
Example: Change AWQ Calibration Settings
```yaml
quant:
  quant_bits: 4
  group_size: 128
  calibration:
    n_samples: 64
    max_seq_length: 1024
```
## 📂 Project Structure
```text
mini-vllm/
├── configs/               # 🌟 YAML configuration files
│   └── default.yaml
├── engine/                # Inference engine & model 
│   └── model_runner.py
├── kernels/               # Custom CUDA/Triton kernels 
├── layers/                # Built-in layer implementations (Attention, MLP, Quantized Linear)
├── model/                 # Model architectures (e.g., Qwen3)
├── utils/                 # Config, Logger, Loader, Quantizer
├── main.py                # Entry point for inference
└── quant.py               # Entry point for quantization model
└── run_quantized.py       # Entry point for inference with quantized model
```
## 🗺️ Roadmap
- [x] **Autoregressive Decoding & KV Cache**: Basic generation loop with Key-Value cache management.
- [x] **AWQ Quantization & Forward Pass**: 4-bit calibration and quantized linear layer forward implementation.
- [x] **Quantized Kernel Support**: Integrate CUDA/Triton kernels for 4-bit model inference.
- [x] **Quantized Model Persistence**: Support saving and loading calibrated AWQ weights.
- [ ] **Implement EOS**: Ensure generation loops break correctly on EOS tokens.
- [ ] **Optimize Quantized Kernel**: Refactor Triton kernels for better performance.
- [ ] **Integrate Automated Profiler**: Embed profiling utils to diagnose performance bottlenecks automatically.

## 🙏 Acknowledgements

This project is inspired by and references the following excellent open-source works or utils:

- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**: For the minimalist and transparent LLM inference architecture design.
- **[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)**: For the robust and efficient AWQ quantization algorithm implementation.
- **[GLM](https://glm-5.org/zh/)**: For the intelligent coding assistance and infrastructure debugging support.

## 📜 License
This project is licensed under the MIT License.