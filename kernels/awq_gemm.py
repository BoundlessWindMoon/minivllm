import torch
import triton
import triton.language as tl


@triton.jit
def _awq_gemm_kernel(
    x_ptr,
    y_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    group_size: tl.constexpr,
    num_pack: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_qn,
    stride_qk,
    stride_sn,
    stride_sk,
    stride_zn,
    stride_zk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    for k_block in range(0, num_k_blocks):
        k_start = k_block * BLOCK_SIZE_K
        offset_k = tl.arange(0, BLOCK_SIZE_K // num_pack)
        offset_sc = tl.arange(0, BLOCK_SIZE_K // group_size)

        qw_ptrs = (
            qweight_ptr
            + offset_n[:, None] * stride_qn
            + (k_start // num_pack + offset_k)[None, :] * stride_qk
        )
        qw = tl.load(qw_ptrs, mask=offset_n[:, None] < N, other=0)

        sc_ptrs = (
            scales_ptr
            + offset_n[:, None] * stride_sn
            + (k_start // group_size + offset_sc)[None, :] * stride_sk
        )
        scale = tl.load(sc_ptrs, mask=offset_n[:, None] < N, other=0.0)

        zero_ptrs = (
            zeros_ptr
            + offset_n[:, None] * stride_zn
            + (k_start // group_size + offset_sc)[None, :] * stride_zk
        )
        zero = tl.load(zero_ptrs, mask=offset_n[:, None] < N, other=0.0)

        for i in tl.static_range(num_pack):
            x_k = k_start + offset_k * num_pack + i
            x_ptrs = x_ptr + offset_m[:, None] * stride_xm + x_k[None, :] * stride_xk
            x_mask = (offset_m[:, None] < M) & (x_k[None, :] < K)
            x_i = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_i = ((qw >> (i * 4)) & 0xF).to(x_i.dtype)
            w_i = (w_i - zero) * scale

            accumulator += tl.dot(x_i, tl.trans(w_i))

    y_ptrs = y_ptr + offset_m[:, None] * N + offset_n[None, :]
    tl.store(
        y_ptrs,
        accumulator,
        mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
    )


def awq_gemm_forward(x, qweight, scales, zeros, group_size, num_pack, bias=None):
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    zeros = zeros.contiguous()

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    M, K = x_2d.shape
    N = qweight.shape[0]

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, group_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _awq_gemm_kernel[grid](
        x_2d,
        y,
        qweight,
        scales,
        zeros,
        M,
        N,
        K,
        group_size,
        num_pack,
        x_2d.stride(0),
        x_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        zeros.stride(0),
        zeros.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )
    if bias is not None:
        y += bias

    final_shape = original_shape[:-1] + (N,)
    return y.reshape(final_shape)
