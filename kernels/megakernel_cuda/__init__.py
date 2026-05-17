"""CUDA megakernel backend — JIT-compiled C++ extension.

Variant dispatch: set ``MINI_VLLM_MK_VARIANT=<key>`` to pick which `.cu` source
is compiled and loaded as the megakernel. Each variant is a separate JIT
extension module (own torch_extensions cache slot) so they don't clobber each
other. All variants must export the same C ABI exposed by ``decode_wrapper.cpp``.
"""

import os
from pathlib import Path

from torch.utils.cpp_extension import load


_KERNEL_DIR = Path(__file__).parent
_VARIANTS_DIR = _KERNEL_DIR / "variants"

# variant key -> list of .cu sources (decode_wrapper.cpp is appended automatically)
VARIANT_SOURCES = {
    "default":      [_KERNEL_DIR / "decode_ldg.cu"],
    "naive":        [_VARIANTS_DIR / "decode_naive.cu"],
    "no_residual":  [_VARIANTS_DIR / "decode_no_residual.cu"],
    "p0":           [_VARIANTS_DIR / "decode_p0.cu"],
    "p1":           [_VARIANTS_DIR / "decode_p1.cu"],
    "p3":           [_VARIANTS_DIR / "decode_p3.cu"],
    "p4":           [_VARIANTS_DIR / "decode_p4.cu"],
    "p6":           [_VARIANTS_DIR / "decode_p6.cu"],
    "p7":           [_VARIANTS_DIR / "decode_p7.cu"],
    "p8":           [_VARIANTS_DIR / "decode_p8.cu"],
    "p9":           [_VARIANTS_DIR / "decode_p9.cu"],
    "p9_combined":  [_VARIANTS_DIR / "decode_p9_combined.cu"],
    "p10":          [_VARIANTS_DIR / "decode_p10.cu"],
    "all_combined": [_VARIANTS_DIR / "decode_all_combined.cu"],
}

_modules: dict[str, object] = {}


def _get_module(variant: str | None = None):
    """Return the JIT-compiled megakernel module for the requested variant.

    Variant resolution order: env var ``MINI_VLLM_MK_VARIANT`` (so ablation
    scripts can override any caller) -> explicit arg -> ``"default"``.
    """
    env_variant = os.environ.get("MINI_VLLM_MK_VARIANT")
    if env_variant:
        variant = env_variant
    elif variant is None:
        variant = "default"

    if variant not in VARIANT_SOURCES:
        raise ValueError(
            f"Unknown megakernel variant {variant!r}. "
            f"Known: {sorted(VARIANT_SOURCES.keys())}"
        )

    if variant in _modules:
        return _modules[variant]

    sources = [str(p) for p in VARIANT_SOURCES[variant]]
    sources.append(str(_KERNEL_DIR / "decode_wrapper.cpp"))

    # Skip variants whose .cu file doesn't exist yet (in-progress ablation work).
    missing = [s for s in sources if not Path(s).exists()]
    if missing:
        raise FileNotFoundError(
            f"Variant {variant!r} sources missing: {missing}"
        )

    _modules[variant] = load(
        name=f"mini_vllm_mk_{variant}",
        sources=sources,
        extra_include_paths=[str(_KERNEL_DIR)],
        extra_cuda_cflags=["-O3", "--generate-line-info"],
        verbose=True,
    )
    return _modules[variant]
