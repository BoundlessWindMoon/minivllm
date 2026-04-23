# kernels/awq_gemm.py
import torch
import triton
import triton.language as tl


@triton.jit
def _awq_gemm_kernel():
    pass


def awq_gemm_forward(x, qweight, scales, zeros, bias=None):
    """
    Triton 融合反量化+矩阵乘法接口
    x: (M, K) fp16
    qweight: (N, K//8) int32
    scales: (N, K//group_size) fp16
    zeros: (N, K//group_size) fp16
    """
    M, K = x.shape
    N = qweight.shape[0]

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _awq_gemm_kernel[grid](
        x,
        qweight,
        scales,
        zeros,
        y,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    if bias is not None:
        y += bias

    return y
