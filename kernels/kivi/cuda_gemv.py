"""JIT-compile the KIVI quantized GEMV CUDA extension on first import."""

import os
import torch
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

_kivi_gemv = load(
    name="kivi_gemv",
    sources=[
        os.path.join(_src_path, "csrc", "pybind.cpp"),
        os.path.join(_src_path, "csrc", "gemv_cuda.cu"),
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
