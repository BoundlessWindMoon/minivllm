"""Triton WT-layout AWQ GEMM kernel."""

import torch
import triton
import triton.language as tl
from utils.logger import logger

_ORIG_CONFIGS = []
_ORIG_BM = [1, 16]
_ORIG_BN = [32, 64, 128]
_ORIG_BK = [64, 128, 256, 512, 1024]
_ORIG_SPLITK = [1, 2, 4]
_ORIG_s = [2]
_ORIG_w = [2]
_MIN_GROUP_SIZE = 128


def _valid_cfg(bk, spk):
    if bk % spk != 0:
        return False
    if bk // spk < 16:
        return False
    if bk > _MIN_GROUP_SIZE:
        return False
    return True


for bm in _ORIG_BM:
    for bn in _ORIG_BN:
        for bk in _ORIG_BK:
            for spk in _ORIG_SPLITK:
                if not _valid_cfg(bk, spk):
                    logger.warning("bk = {bk}, spk = {spk}, config is not supported !")
                    continue
                for s in _ORIG_s:
                    for w in _ORIG_w:
                        _ORIG_CONFIGS.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_M": bm,
                                    "BLOCK_SIZE_N": bn,
                                    "BLOCK_SIZE_K": bk,
                                    "SPLIT_K": spk,
                                },
                                num_stages=s,
                                num_warps=w,
                            )
                        )


@triton.autotune(
    configs=_ORIG_CONFIGS,
    key=["M", "N", "K"],
    reset_to_zero=["c_ptr"],
    cache_results=True,
    warmup=25,
    rep=100,
)
@triton.jit
def _awq_gemm_kernel_wt(
    a_ptr,
    c_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    M,
    N,
    K,
    group_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_qk,
    stride_qn,
    stride_sk,
    stride_sn,
    stride_zk,
    stride_zn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # unpack pid_m, pid_n, pid_k
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    BSK_LOCAL: tl.constexpr = BLOCK_SIZE_K // SPLIT_K

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k_local = pid_z * BSK_LOCAL + tl.arange(0, BSK_LOCAL)
    offset_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    shifts = (tl.arange(0, BLOCK_SIZE_N) % 8) * 4
    shifts = tl.reshape(shifts, (1, BLOCK_SIZE_N))

    num_k_steps = tl.cdiv(K, BLOCK_SIZE_K)

    for k_step in range(num_k_steps):
        k_block_start = k_step * BLOCK_SIZE_K
        offset_k = k_block_start + offset_k_local

        a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
        a_mask = (offset_m[:, None] < M) & (offset_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = (
            qweight_ptr + offset_k[:, None] * stride_qk + offset_bn[None, :] * stride_qn
        )
        b_mask = (offset_k[:, None] < K) & (offset_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0)

        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = ((b >> shifts) & 0xF).to(tl.float32)

        k_gs = k_block_start // group_size
        offset_sc = k_gs + tl.arange(0, 1)

        sc_ptrs = (
            scales_ptr + offset_sc[:, None] * stride_sk + offset_n[None, :] * stride_sn
        )
        sc_mask = (offset_sc[:, None] < K // group_size) & (offset_n[None, :] < N)
        sc = tl.load(sc_ptrs, mask=sc_mask, other=0.0)
        sc = tl.broadcast_to(sc, (BSK_LOCAL, BLOCK_SIZE_N))

        zr_ptrs = (
            zeros_ptr + offset_sc[:, None] * stride_zk + offset_n[None, :] * stride_zn
        )
        zr_mask = (offset_sc[:, None] < K // group_size) & (offset_n[None, :] < N)
        zr = tl.load(zr_ptrs, mask=zr_mask, other=0.0)
        zr = tl.broadcast_to(zr, (BSK_LOCAL, BLOCK_SIZE_N))

        b = (b - zr) * sc
        b = b.to(a.dtype)

        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

    c_ptrs = c_ptr + offset_m[:, None] * N + offset_n[None, :]
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator.to(c_ptr.type.element_ty), mask=c_mask)


def awq_gemm_forward_wt(x, qweight, scales, zeros, group_size, num_pack, bias=None):
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    M, K = x_2d.shape
    N = scales.shape[1]

    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        meta["SPLIT_K"],
    )

    _awq_gemm_kernel_wt[grid](
        x_2d,
        y,
        qweight,
        scales,
        zeros,
        M,
        N,
        K,
        group_size,
        x_2d.stride(0),
        x_2d.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        zeros.stride(0),
        zeros.stride(1),
    )

    if bias is not None:
        y += bias

    final_shape = original_shape[:-1] + (N,)
    return y.reshape(final_shape)
