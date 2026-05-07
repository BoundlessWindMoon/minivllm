"""CUDA megakernel backend — JIT-compiled C++ extension."""

from pathlib import Path
from torch.utils.cpp_extension import load

_KERNEL_DIR = Path(__file__).parent

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load(
            name="mini_vllm_megakernel_cuda",
            sources=[
                str(_KERNEL_DIR / "decode_ldg.cu"),
                str(_KERNEL_DIR / "decode_wrapper.cpp"),
            ],
            extra_include_paths=[str(_KERNEL_DIR)],
            extra_cuda_cflags=["-O3", "--generate-line-info"],
            verbose=True,
        )
    return _module
