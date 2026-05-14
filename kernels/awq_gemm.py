"""GEMM backend for AWQ 4-bit inference."""

import torch
import triton
import triton.language as tl

_ORIG_CONFIGS = []
for bm, bn, sw in [
    (1, 32, [(1, 1), (1, 2), (2, 2)]),
    (1, 64, [(1, 1), (1, 2), (2, 2)]),
    (1, 128, [(1, 1), (1, 2), (2, 2)]),
    (16, 16, [(1, 1), (1, 2), (2, 2)]),
    (16, 32, [(1, 1), (1, 2), (2, 2)]),
    (16, 64, [(1, 1), (1, 2), (2, 2)]),
    (16, 128, [(1, 1), (1, 2), (2, 2), (2, 4)]),
]:
    for s, w in sw:
        _ORIG_CONFIGS.append(
            triton.Config(
                {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn},
                num_stages=s,
                num_warps=w,
            )
        )

_UNROLL_CONFIGS = []
for bm, bn, sw, ufs in [
    (16, 64, [(1, 1), (1, 2), (2, 2)], [1, 8]),
    (16, 128, [(1, 2), (2, 2), (2, 4)], [1, 8]),
]:
    for s, w in sw:
        for uf in ufs:
            _UNROLL_CONFIGS.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_N": bn,
                        "UNROLL_FACTOR": uf,
                    },
                    num_stages=s,
                    num_warps=w,
                )
            )

_FUSED_CONFIGS = []
for bm, bn, sw, ufs, kgp in [
    (16, 64, [(1, 1), (2, 2)], [1, 4], [1, 2]),
    (16, 128, [(1, 1), (2, 2)], [1, 4], [1, 2, 4]),
]:
    for s, w in sw:
        for uf in ufs:
            for k in kgp:
                _FUSED_CONFIGS.append(
                    triton.Config(
                        {
                            "BLOCK_SIZE_M": bm,
                            "BLOCK_SIZE_N": bn,
                            "UNROLL_FACTOR": uf,
                            "K_GROUP": k,
                        },
                        num_stages=s,
                        num_warps=w,
                    )
                )

_SPLITK_CONFIGS = []
for bm, bn, sw, kgp in [
    (16, 64, [(1, 1), (1, 2), (2, 2)], [1, 2, 4]),
    (16, 128, [(1, 1), (1, 2), (2, 2), (2, 4)], [1, 2, 4]),
]:
    for s, w in sw:
        for k in kgp:
            _SPLITK_CONFIGS.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": bm,
                        "BLOCK_SIZE_N": bn,
                        "K_GROUP": k,
                    },
                    num_stages=s,
                    num_warps=w,
                )
            )


@triton.autotune(
    configs=_ORIG_CONFIGS, key=["M", "N", "K"], cache_results=True, warmup=25, rep=100
)
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

        x_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for i in tl.static_range(num_pack):
            x_k = k_start + offset_k * num_pack + i
            x_ptrs = x_ptr + offset_m[:, None] * stride_xm + x_k[None, :] * stride_xk
            x_mask = (offset_m[:, None] < M) & (x_k[None, :] < K)
            x_i = tl.load(x_ptrs, mask=x_mask, other=0.0)

            x_sum += tl.sum(x_i, axis=1).to(tl.float32)

            w_i = ((qw >> (i * 4)) & 0xF).to(x_i.dtype)
            w_i = w_i * scale

            accumulator += tl.dot(x_i, tl.trans(w_i))

        accumulator -= x_sum[:, None] * tl.trans(zero * scale)

    y_ptrs = y_ptr + offset_m[:, None] * N + offset_n[None, :]
    tl.store(
        y_ptrs,
        accumulator,
        mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
    )


@triton.autotune(
    configs=_SPLITK_CONFIGS, key=["M", "N", "K"], cache_results=True, warmup=25, rep=100
)
@triton.jit
def _awq_gemm_kernel_splitk(
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
    K_GROUP: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    k_blocks_per_pid = tl.cdiv(num_k_blocks, K_GROUP)
    k_start_block = pid_k * k_blocks_per_pid
    k_end_block = tl.minimum(k_start_block + k_blocks_per_pid, num_k_blocks)
    for k_block in range(k_start_block, k_end_block):
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

        x_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for i in tl.static_range(num_pack):
            x_k = k_start + offset_k * num_pack + i
            x_ptrs = x_ptr + offset_m[:, None] * stride_xm + x_k[None, :] * stride_xk
            x_mask = (offset_m[:, None] < M) & (x_k[None, :] < K)
            x_i = tl.load(x_ptrs, mask=x_mask, other=0.0)

            x_sum += tl.sum(x_i, axis=1).to(tl.float32)

            w_i = ((qw >> (i * 4)) & 0xF).to(x_i.dtype)
            w_i = w_i * scale

            accumulator += tl.dot(x_i, tl.trans(w_i))

        accumulator -= x_sum[:, None] * tl.trans(zero * scale)

    y_ptrs = y_ptr + offset_m[:, None] * N + offset_n[None, :]

    if K_GROUP == 1:
        tl.store(
            y_ptrs,
            accumulator,
            mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
        )
    else:
        tl.atomic_add(
            y_ptrs,
            accumulator,
            mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
        )


@triton.autotune(
    configs=_UNROLL_CONFIGS, key=["M", "N", "K"], cache_results=True, warmup=25, rep=100
)
@triton.jit
def _awq_gemm_kernel_unroll(
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
    UNROLL_FACTOR: tl.constexpr,
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

        x_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for i_group in tl.static_range(0, num_pack, UNROLL_FACTOR):
            x_k_offsets = (
                k_start
                + offset_k[:, None] * num_pack
                + (i_group + tl.arange(0, UNROLL_FACTOR))[None, :]
            )

            x_k_flat = x_k_offsets.reshape(BLOCK_SIZE_K // num_pack * UNROLL_FACTOR)
            x_ptrs = (
                x_ptr + offset_m[:, None] * stride_xm + x_k_flat[None, :] * stride_xk
            )
            x_mask = (offset_m[:, None] < M) & (x_k_flat[None, :] < K)

            x_group = tl.load(x_ptrs, mask=x_mask, other=0.0)

            x_sum += tl.sum(x_group, axis=1).to(tl.float32)

            shifts = (i_group + tl.arange(0, UNROLL_FACTOR)) * 4

            w_3d = ((qw[:, :, None] >> shifts[None, None, :]) & 0xF).to(x_group.dtype)

            w_group = w_3d.reshape(
                BLOCK_SIZE_N, BLOCK_SIZE_K // num_pack * UNROLL_FACTOR
            )

            w_group = w_group * scale

            accumulator += tl.dot(x_group, tl.trans(w_group))

        accumulator -= x_sum[:, None] * tl.trans(zero * scale)
    y_ptrs = y_ptr + offset_m[:, None] * N + offset_n[None, :]
    tl.store(
        y_ptrs,
        accumulator,
        mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
    )


@triton.autotune(
    configs=_FUSED_CONFIGS, key=["M", "N", "K"], cache_results=True, warmup=25, rep=100
)
@triton.jit
def _awq_gemm_kernel_fused(
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
    UNROLL_FACTOR: tl.constexpr,
    K_GROUP: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    k_blocks_per_pid = tl.cdiv(num_k_blocks, K_GROUP)
    k_start_block = pid_k * k_blocks_per_pid
    k_end_block = tl.minimum(k_start_block + k_blocks_per_pid, num_k_blocks)

    for k_block in range(k_start_block, k_end_block):
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

        x_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for i_group in tl.static_range(0, num_pack, UNROLL_FACTOR):
            x_k_offsets = (
                k_start
                + offset_k[:, None] * num_pack
                + (i_group + tl.arange(0, UNROLL_FACTOR))[None, :]
            )

            x_k_flat = x_k_offsets.reshape(BLOCK_SIZE_K // num_pack * UNROLL_FACTOR)
            x_ptrs = (
                x_ptr + offset_m[:, None] * stride_xm + x_k_flat[None, :] * stride_xk
            )
            x_mask = (offset_m[:, None] < M) & (x_k_flat[None, :] < K)

            x_group = tl.load(x_ptrs, mask=x_mask, other=0.0)

            x_sum += tl.sum(x_group, axis=1).to(tl.float32)

            shifts = (i_group + tl.arange(0, UNROLL_FACTOR)) * 4

            w_3d = ((qw[:, :, None] >> shifts[None, None, :]) & 0xF).to(x_group.dtype)

            w_group = w_3d.reshape(
                BLOCK_SIZE_N, BLOCK_SIZE_K // num_pack * UNROLL_FACTOR
            )

            w_group = w_group * scale

            accumulator += tl.dot(x_group, tl.trans(w_group))

        accumulator -= x_sum[:, None] * tl.trans(zero * scale)

    y_ptrs = y_ptr + offset_m[:, None] * N + offset_n[None, :]
    if K_GROUP == 1:
        tl.store(
            y_ptrs,
            accumulator,
            mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
        )
    else:
        tl.atomic_add(
            y_ptrs,
            accumulator,
            mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
        )


def awq_gemm_forward(
    x, qweight, scales, zeros, group_size, num_pack, bias=None, version="triton_naive"
):

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    M, K = x_2d.shape
    N = qweight.shape[0]

    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    BLOCK_K = group_size

    PREFILL_M_THRESHOLD = 32

    if 'unroll' in version:
        kernel = _awq_gemm_kernel_unroll
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
        kernel[grid](
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
            BLOCK_SIZE_K=BLOCK_K,
        )
    elif 'fused' in version:
        kernel = _awq_gemm_kernel_fused
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            meta["K_GROUP"],
        )
        kernel[grid](
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
            BLOCK_SIZE_K=BLOCK_K,
        )
    elif 'splitk' in version:
        kernel = _awq_gemm_kernel_splitk
        # Decode – let autotune pick the best tile size.
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            meta["K_GROUP"],
        )
        kernel[grid](
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
            BLOCK_SIZE_K=BLOCK_K,
        )
    else:
        kernel = _awq_gemm_kernel

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
        kernel[grid](
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
            BLOCK_SIZE_K=BLOCK_K,
        )

    if bias is not None:
        y += bias

    final_shape = original_shape[:-1] + (N,)
    return y.reshape(final_shape)
