"""Python wrappers for KIVI quantized BMM CUDA kernels.

Adapts the reference KIVI implementation to mini-vllm's data layout.
"""

import torch
from kernels.kivi.gemv_cuda import _kivi_gemv


def kivi_bmm_fA_qB_outer(
    group_size: int,
    fA: torch.Tensor,
    qB: torch.Tensor,
    scales: torch.Tensor,
    mn: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Quantized-domain batched matrix multiplication.

    Computes C = A x B where B is asymmetrically quantized and packed.
    Supports GQA via nh / nh_kv head replication inside the CUDA kernel.

    Args:
        group_size: quantization group size.
        fA: float16 tensor of shape (B, nh, M, K).
        qB: int32 packed tensor of shape (B, nh_kv, K, N // feat_per_int).
        scales: float16 tensor of shape (B, nh_kv, K, N // group_size).
        mn: float16 tensor of shape (B, nh_kv, K, N // group_size).
            Minimum value per group (KIVI uses min, not zero-point).
        bits: 2 or 4.

    Returns:
        float16 tensor of shape (B, nh, M, N).
    """
    assert len(fA.shape) == 4 and len(qB.shape) == 4
    B, nh, M, K = fA.shape
    nh_kv = qB.shape[1]
    feat_per_int = 32 // bits

    # CUDA kernel only supports fp16; up/down-cast from bf16 when necessary
    orig_dtype = fA.dtype
    if orig_dtype == torch.bfloat16:
        fA = fA.half()
        scales = scales.half()
        mn = mn.half()

    fA = fA.view(-1, M, K).contiguous()
    N = qB.shape[-1] * feat_per_int
    qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
    flatten_B = B * nh_kv
    scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    mn = mn.view(flatten_B, mn.shape[-2], mn.shape[-1]).transpose(1, 2).contiguous()

    assert bits in (2, 4)
    assert nh % nh_kv == 0
    if group_size not in (64, 128):
        raise ValueError(
            f"CUDA GEMV kernel only supports group_size 64 or 128, got {group_size}. "
            f"Use the Triton fallback for other group sizes."
        )

    c = _kivi_gemv.gemv_forward_cuda_outer_dim(
        fA, qB, scales, mn, bits, group_size, nh, nh_kv
    )
    c = c.view(B, nh, M, N)
    if orig_dtype == torch.bfloat16:
        c = c.bfloat16()
    return c
