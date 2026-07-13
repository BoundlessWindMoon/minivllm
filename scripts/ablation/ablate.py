"""Stable ablation FIXED: no autotune for ANY kernel, fixed config for all.

This fixes the contamination in ablate_stable.py where awq_gemm_forward_wt_fused
still had autotune (48 configs, ~6000 calls to complete) while custom kernels
were fixed-config. The autotune pollution made awq_fused appear ~70us slower
than its actual kernel execution time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import subprocess
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Best-effort GPU clock locking
# ---------------------------------------------------------------------------
def _lock_gpu_clocks():
    try:
        subprocess.run(
            ["nvidia-smi", "-lgc", "1500"],
            capture_output=True,
            check=False,
        )
        print("[INFO] Attempted GPU clock lock to 1500 MHz")
    except Exception as e:
        print(f"[WARN] Could not lock GPU clocks: {e}")


def _unlock_gpu_clocks():
    try:
        subprocess.run(
            ["nvidia-smi", "-rgc"],
            capture_output=True,
            check=False,
        )
        print("[INFO] Reset GPU clocks")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Fixed config for ALL kernels (no autotune)
# ---------------------------------------------------------------------------
_FIXED_M = tl.constexpr(1)
_FIXED_N = tl.constexpr(64)
_FIXED_K = tl.constexpr(128)
_FIXED_SPLITK = tl.constexpr(1)


# ---------------------------------------------------------------------------
# Kernel 0: simplest_triton — canonical matmul, no group loop
# ---------------------------------------------------------------------------
@triton.jit
def _simplest_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NAME: tl.constexpr = "my_simple",
):
    BLOCK_M: tl.constexpr = _FIXED_M
    BLOCK_N: tl.constexpr = _FIXED_N
    BLOCK_K: tl.constexpr = _FIXED_K
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def simplest_triton_forward(x, weight):
    orig = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = weight.shape[1]
    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, 1) * triton.cdiv(N, 64),)
    _simplest_matmul_kernel[grid](
        x2,
        weight,
        y,
        M,
        N,
        K,
        x2.stride(0),
        x2.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
    )
    return y.reshape(orig[:-1] + (N,))


# ---------------------------------------------------------------------------
# Kernel 1: fp16_triton — same group-loop structure as AWQ
# ---------------------------------------------------------------------------
@triton.jit
def _fp16_triton_kernel(
    a_ptr,
    c_ptr,
    weight_ptr,
    M,
    N,
    K,
    group_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    NAME: tl.constexpr = "my_fp16",
):
    BLOCK_SIZE_M: tl.constexpr = _FIXED_M
    BLOCK_SIZE_N: tl.constexpr = _FIXED_N
    BLOCK_SIZE_K: tl.constexpr = _FIXED_K
    SPLIT_K: tl.constexpr = _FIXED_SPLITK
    pid = tl.program_id(0)
    pid_z = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    BSK_LOCAL: tl.constexpr = BLOCK_SIZE_K // SPLIT_K
    BK_PER_GROUP: tl.constexpr = group_size // BLOCK_SIZE_K
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k_local = pid_z * BSK_LOCAL + tl.arange(0, BSK_LOCAL)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_groups = tl.cdiv(K, group_size)
    for g_step in range(num_groups):
        for j in range(BK_PER_GROUP):
            k_step = g_step * BK_PER_GROUP + j
            k_block_start = k_step * BLOCK_SIZE_K
            offset_k = k_block_start + offset_k_local
            a = tl.load(
                a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak,
                mask=(offset_m[:, None] < M) & (offset_k[None, :] < K),
                other=0.0,
            )
            w = tl.load(
                weight_ptr
                + offset_k[:, None] * stride_wk
                + offset_n[None, :] * stride_wn,
                mask=(offset_k[:, None] < K) & (offset_n[None, :] < N),
                other=0.0,
            )
            accumulator = tl.dot(a, w, accumulator, out_dtype=tl.float32)
    c_ptrs = c_ptr + offset_m[:, None] * N + offset_n[None, :]
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fp16_triton_forward(x, weight, group_size):
    orig = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = weight.shape[1]
    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, 1) * triton.cdiv(N, 64), 1)
    _fp16_triton_kernel[grid](
        x2,
        y,
        weight,
        M,
        N,
        K,
        group_size,
        x2.stride(0),
        x2.stride(1),
        weight.stride(0),
        weight.stride(1),
    )
    return y.reshape(orig[:-1] + (N,))


# ---------------------------------------------------------------------------
# Kernel 2: awq_unpack — unpack int4 -> fp16, NO scale/zero
# ---------------------------------------------------------------------------
@triton.jit
def _awq_unpack_kernel(
    a_ptr,
    c_ptr,
    qweight_ptr,
    M,
    N,
    K,
    group_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_qk,
    stride_qn,
    NAME: tl.constexpr = "my_unpack",
):
    BLOCK_SIZE_M: tl.constexpr = _FIXED_M
    BLOCK_SIZE_N: tl.constexpr = _FIXED_N
    BLOCK_SIZE_K: tl.constexpr = _FIXED_K
    SPLIT_K: tl.constexpr = _FIXED_SPLITK
    pid = tl.program_id(0)
    pid_z = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    BSK_LOCAL: tl.constexpr = BLOCK_SIZE_K // SPLIT_K
    BK_PER_GROUP: tl.constexpr = group_size // BLOCK_SIZE_K
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k_local = pid_z * BSK_LOCAL + tl.arange(0, BSK_LOCAL)
    offset_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    shifts = (tl.arange(0, BLOCK_SIZE_N) % 8) * 4
    shifts = tl.reshape(shifts, (1, BLOCK_SIZE_N))
    num_groups = tl.cdiv(K, group_size)
    for g_step in range(num_groups):
        for j in range(BK_PER_GROUP):
            k_step = g_step * BK_PER_GROUP + j
            k_block_start = k_step * BLOCK_SIZE_K
            offset_k = k_block_start + offset_k_local
            a = tl.load(
                a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak,
                mask=(offset_m[:, None] < M) & (offset_k[None, :] < K),
                other=0.0,
            )
            b_ptrs = (
                qweight_ptr
                + offset_k[:, None] * stride_qk
                + offset_bn[None, :] * stride_qn
            )
            b_mask = (offset_k[:, None] < K) & (offset_bn[None, :] < N // 8)
            b_packed = tl.load(b_ptrs, mask=b_mask, other=0)
            b = tl.interleave(b_packed, b_packed)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = ((b >> shifts) & 0xF).to(tl.float32)
            b = b.to(a.dtype)
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
    c_ptrs = c_ptr + offset_m[:, None] * N + offset_n[None, :]
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def awq_unpack_forward(x, qweight, group_size):
    orig = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = qweight.shape[1] * 8
    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, 1) * triton.cdiv(N, 64), 1)
    _awq_unpack_kernel[grid](
        x2,
        y,
        qweight,
        M,
        N,
        K,
        group_size,
        x2.stride(0),
        x2.stride(1),
        qweight.stride(0),
        qweight.stride(1),
    )
    return y.reshape(orig[:-1] + (N,))


# ---------------------------------------------------------------------------
# Kernel 3: awq_fused FIXED — no autotune, same fixed config
# ---------------------------------------------------------------------------
@triton.jit
def _awq_gemm_kernel_wt_fused_fixed(
    a_ptr,
    c_ptr,
    qweight_ptr,
    scales_ptr,
    zero_scales_ptr,
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
    stride_zsk,
    stride_zsn,
    NAME: tl.constexpr = "my_fused",
):
    BLOCK_SIZE_M: tl.constexpr = _FIXED_M
    BLOCK_SIZE_N: tl.constexpr = _FIXED_N
    BLOCK_SIZE_K: tl.constexpr = _FIXED_K
    SPLIT_K: tl.constexpr = _FIXED_SPLITK

    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    BSK_LOCAL: tl.constexpr = BLOCK_SIZE_K // SPLIT_K
    BK_PER_GROUP: tl.constexpr = group_size // BLOCK_SIZE_K

    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k_local = pid_z * BSK_LOCAL + tl.arange(0, BSK_LOCAL)
    offset_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    shifts = (tl.arange(0, BLOCK_SIZE_N) % 8) * 4
    shifts = tl.reshape(shifts, (1, BLOCK_SIZE_N))
    num_groups = tl.cdiv(K, group_size)

    for g_step in range(num_groups):
        offset_kg = g_step + tl.arange(0, 1)
        sc_mask = (offset_kg[:, None] < K // group_size) & (offset_n[None, :] < N)

        sc_ptrs = (
            scales_ptr + offset_kg[:, None] * stride_sk + offset_n[None, :] * stride_sn
        )
        sc = tl.load(sc_ptrs, mask=sc_mask, other=0.0)
        sc = tl.broadcast_to(sc, (BSK_LOCAL, BLOCK_SIZE_N))

        zs_ptrs = (
            zero_scales_ptr
            + offset_kg[:, None] * stride_zsk
            + offset_n[None, :] * stride_zsn
        )
        zs = tl.load(zs_ptrs, mask=sc_mask, other=0.0)
        zs = tl.broadcast_to(zs, (BSK_LOCAL, BLOCK_SIZE_N))

        for j in range(BK_PER_GROUP):
            k_step = g_step * BK_PER_GROUP + j
            k_block_start = k_step * BLOCK_SIZE_K
            offset_k = k_block_start + offset_k_local

            a_ptrs = (
                a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
            )
            a_mask = (offset_m[:, None] < M) & (offset_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            b_ptrs = (
                qweight_ptr
                + offset_k[:, None] * stride_qk
                + offset_bn[None, :] * stride_qn
            )
            b_mask = (offset_k[:, None] < K) & (offset_bn[None, :] < N // 8)
            b_packed = tl.load(b_ptrs, mask=b_mask, other=0)

            b = tl.interleave(b_packed, b_packed)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = ((b >> shifts) & 0xF).to(tl.float32)
            b = b * sc - zs
            b = b.to(a.dtype)

            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

    c_ptrs = c_ptr + offset_m[:, None] * N + offset_n[None, :]
    c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def awq_fused_fixed_forward(x, qweight, scales, zero_scales, group_size):
    orig = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = scales.shape[1]
    y = torch.zeros((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, 1) * triton.cdiv(N, 64), 1)
    _awq_gemm_kernel_wt_fused_fixed[grid](
        x2,
        y,
        qweight,
        scales,
        zero_scales,
        M,
        N,
        K,
        group_size,
        x2.stride(0),
        x2.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        zero_scales.stride(0),
        zero_scales.stride(1),
    )
    return y.reshape(orig[:-1] + (N,))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def _prepare_bufs(M, K, N, group_size, dev="cuda"):
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    qweight = torch.randint(0, 2**31, (K, N // 8), dtype=torch.int32, device=dev)
    scales = torch.randn(K // group_size, N, dtype=torch.float16, device=dev)
    unpack_zeros = torch.randn(K // group_size, N, dtype=torch.float16, device=dev)
    zero_scales = (unpack_zeros * scales).half()
    weight_fp16 = torch.randn(K, N, dtype=torch.float16, device=dev)
    return x, qweight, scales, zero_scales, weight_fp16


def _time_once(fn, iters, *args):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def _bench_kernel(fn, warmup_iters, bench_iters, measure_runs, *args):
    for _ in range(warmup_iters):
        _ = fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(measure_runs):
        t = _time_once(fn, bench_iters, *args)
        times.append(t)
        torch.cuda.synchronize()

    times.sort()
    median = times[len(times) // 2]
    mad = sum(abs(t - median) for t in times) / len(times)
    return median, mad, times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _lock_gpu_clocks()
    try:
        group_size = 128
        layers = [
            {"name": "qkv_proj", "K": 1024, "N": 4096},
            {"name": "o_proj", "K": 2048, "N": 1024},
            {"name": "gate_up_proj", "K": 1024, "N": 6144},
            {"name": "down_proj", "K": 3072, "N": 1024},
        ]
        batch_sizes = [1]

        WARMUP = 50
        BENCH_ITERS = 1000
        MEASURE_RUNS = 10

        print("=" * 130)
        print("Stable AWQ Ablation FIXED — ALL kernels use fixed config, NO autotune")
        print(
            f"Warmup={WARMUP}, iters={BENCH_ITERS}, fixed BLOCK_M=1, BLOCK_N=64, BLOCK_K=128, SPLIT_K=1"
        )
        print("=" * 130)

        for M in batch_sizes:
            print(f"\n--- M = {M} ---")
            print(
                f"{'Layer':>15s} {'K':>6s} {'N':>6s} "
                f"{'cuBLAS':>12s} {'simplest':>12s} {'triton_fp16':>13s} {'unpack':>12s} {'awq_fused':>12s} "
                f"{'cu/simp':>8s} {'unpack%':>8s} {'dequant%':>9s}"
            )
            print("-" * 125)

            for layer in layers:
                name, K, N = layer["name"], layer["K"], layer["N"]
                x, qw, sc, zs, wfp = _prepare_bufs(M, K, N, group_size)

                t_cublas, mad_c, _ = _bench_kernel(
                    lambda: torch.matmul(x, wfp), WARMUP, BENCH_ITERS, MEASURE_RUNS
                )
                t_simplest, mad_s, _ = _bench_kernel(
                    simplest_triton_forward, WARMUP, BENCH_ITERS, MEASURE_RUNS, x, wfp
                )
                t_fp16, mad_f, _ = _bench_kernel(
                    fp16_triton_forward,
                    WARMUP,
                    BENCH_ITERS,
                    MEASURE_RUNS,
                    x,
                    wfp,
                    group_size,
                )
                t_unpack, mad_u, _ = _bench_kernel(
                    awq_unpack_forward,
                    WARMUP,
                    BENCH_ITERS,
                    MEASURE_RUNS,
                    x,
                    qw,
                    group_size,
                )
                t_fused, mad_d, _ = _bench_kernel(
                    awq_fused_fixed_forward,
                    WARMUP,
                    BENCH_ITERS,
                    MEASURE_RUNS,
                    x,
                    qw,
                    sc,
                    zs,
                    group_size,
                )

                ratio_cu_simp = t_cublas / t_simplest
                unpack_pct = (t_unpack - t_fp16) / t_fp16 * 100
                dequant_pct = (t_fused - t_unpack) / t_fp16 * 100

                print(
                    f"{name:>15s} {K:>6d} {N:>6d} "
                    f"{t_cublas:>11.1f}±{mad_c:>3.1f} {t_simplest:>11.1f}±{mad_s:>3.1f} "
                    f"{t_fp16:>12.1f}±{mad_f:>3.1f} {t_unpack:>11.1f}±{mad_u:>3.1f} {t_fused:>11.1f}±{mad_d:>3.1f} "
                    f"{ratio_cu_simp:>8.2f}x {unpack_pct:>8.1f} {dequant_pct:>9.1f}"
                )

        print("\n" + "=" * 130)
        print("Legend:")
        print("  cuBLAS      = torch.matmul(x, fp16_weight)")
        print("  simplest    = Tutorial-style Triton matmul (no group loop)")
        print("  triton_fp16 = Triton with same group-loop structure as AWQ")
        print("  unpack      = Triton unpacks int4 -> fp16, no scale/zero")
        print("  awq_fused   = Full kernel (unpack + hoisted scale + FMA), NO AUTOTUNE")
        print("  ±X          = mean absolute deviation from median")
        print(
            "\nNOTE: This script uses a FIXED-CONFIG awq_fused kernel to eliminate autotune pollution."
        )
        print(
            "      The original awq_gemm_forward_wt_fused has 48 autotune configs and needs ~6000"
        )
        print(
            "      warmup calls to complete search. Without that, it appears ~70us slower than it is."
        )
        print("=" * 130)
    finally:
        _unlock_gpu_clocks()


if __name__ == "__main__":
    main()
