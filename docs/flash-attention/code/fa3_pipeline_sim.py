"""
fa3_pipeline_sim.py  —  FA3 Warp Specialization + GEMM-Softmax 重叠模拟

用 Python 线程模拟 Producer WG（load）和 Consumer WG（MMA+softmax）
的 pipeline 行为，演示两阶段 softmax 如何与 GEMM 重叠。

依赖：torch >= 2.0
可选：flash_attn（GPU 性能对比）

用法：
    python fa3_pipeline_sim.py
"""

import math
import time
import threading
import queue
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 参考实现
# ──────────────────────────────────────────────────────────────
def attention_reference(Q, K, V, scale, causal=False):
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    return torch.matmul(torch.softmax(S, dim=-1), V)


# ──────────────────────────────────────────────────────────────
# FA2 风格：max+exp+sum 合并在一步（无法与 GEMM 重叠）
# ──────────────────────────────────────────────────────────────
def fa2_style_softmax_step(s, m_prev, l_prev, o_prev, vj, scale):
    """
    对应 FA2 softmax_rescale_o：max、exp、sum、rescale-O 全部连续执行。
    GEMM(S_t) 结束 → softmax(S_t) → rescale(O) → GEMM(S_{t+1})
    没有重叠机会。
    """
    # 阶段 A + B 合并
    m_new   = torch.maximum(m_prev, s.max(dim=1).values)
    alpha   = torch.exp(m_prev - m_new)
    p       = torch.exp(s - m_new.unsqueeze(1))
    l_new   = alpha * l_prev + p.sum(dim=1)
    o_new   = alpha.unsqueeze(1) * o_prev + torch.matmul(p, vj)
    return m_new, l_new, o_new


# ──────────────────────────────────────────────────────────────
# FA3 风格：两阶段 softmax（允许与下一次 GEMM 重叠）
# ──────────────────────────────────────────────────────────────
def fa3_phase_A(s, m_prev, l_prev):
    """
    阶段 A：max_get_scale
    只计算 m^(t) 和 α^(t)，立即 rescale l（但 O 的 rescale 可推迟）。
    可以在 WGMMA(S_{t+1}) 启动后执行。
    """
    m_new  = torch.maximum(m_prev, s.max(dim=1).values)
    alpha  = torch.exp(m_prev - m_new)           # rescale 系数
    l_new  = alpha * l_prev                       # l 先 rescale，sum 留到阶段 B
    return m_new, alpha, l_new


def fa3_phase_B(s, m_new, l_partial, o_prev, alpha, vj):
    """
    阶段 B：online_softmax + O 更新
    在 WGMMA 结果就绪后执行 exp、sum、O 累积。
    """
    p      = torch.exp(s - m_new.unsqueeze(1))   # p = exp(S_t - m^(t))
    l_new  = l_partial + p.sum(dim=1)            # 补上本 block 的 sum
    o_new  = alpha.unsqueeze(1) * o_prev + torch.matmul(p, vj)
    return l_new, o_new


# ──────────────────────────────────────────────────────────────
# 正确性验证：FA3 两阶段 = FA2 结果
# ──────────────────────────────────────────────────────────────
def flash_attention_fa3_style(Q, K, V, scale=None, causal=False, Br=64, Bc=64):
    """
    使用 FA3 两阶段 softmax 的 tiling attention。

    关键：两阶段和一阶段数学等价，区别只在执行时序上。
    阶段 A（max_get_scale）：计算 m^(t), α^(t)，立即 rescale O
    阶段 B（online_softmax）：计算 p, 更新 l, 累积 p@V

    正确的递推（注意 O 在 alpha rescale 后再加 p@V）：
        alpha = exp(m_prev - m_new)
        o     = alpha * o + exp(s - m_new) @ vj   ← 两步合一也正确
    """
    B, H, N, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    O_out = torch.zeros_like(Q)

    for b in range(B):
        for h in range(H):
            q, k, v = Q[b, h], K[b, h], V[b, h]
            for i0 in range(0, N, Br):
                i1 = min(i0 + Br, N)
                qi = q[i0:i1]
                br = i1 - i0

                m = torch.full((br,), float('-inf'), dtype=Q.dtype)
                l = torch.zeros(br, dtype=Q.dtype)
                o = torch.zeros(br, d, dtype=Q.dtype)

                for j0 in range(0, N, Bc):
                    j1 = min(j0 + Bc, N)
                    kj = k[j0:j1]
                    vj = v[j0:j1]
                    s  = torch.matmul(qi, kj.T) * scale
                    if causal:
                        rows = torch.arange(i0, i1).unsqueeze(1)
                        cols = torch.arange(j0, j1).unsqueeze(0)
                        s = s.masked_fill(cols > rows, float('-inf'))

                    # ── 阶段 A：max_get_scale ─────────────────────
                    # 在 FA3 kernel 里，这步在下一次 WGMMA 启动后立即执行
                    m_new, alpha, l_rescaled = fa3_phase_A(s, m, l)
                    m = m_new

                    # ── 阶段 A 结束后立即 rescale O ───────────────
                    # 对应 FA3 Consumer WG：scores_scale 返回后即 rescale acc_o
                    o = alpha.unsqueeze(1) * o

                    # ── 阶段 B：online_softmax，等 WGMMA 完成 ─────
                    p     = torch.exp(s - m.unsqueeze(1))           # exp(S_t - m^(t))
                    l     = l_rescaled + p.sum(dim=1)               # l^(t)
                    o     = o + torch.matmul(p, vj)                 # O^(t)（已 rescale）

                O_out[b, h, i0:i1] = o / l.unsqueeze(1)
    return O_out


# ──────────────────────────────────────────────────────────────
# Pipeline 重叠时间收益模拟（用 sleep 模拟各操作耗时）
# ──────────────────────────────────────────────────────────────
def simulate_pipeline_overlap(T=8, t_gemm=1.0, t_softmax=0.3, t_load=0.8):
    """
    模拟 T 个 KV block 的处理时间。

    FA2：串行 load → GEMM → softmax（无重叠）
    FA3：Producer load 与 Consumer GEMM+softmax 并行（重叠）

    参数：
        T        : KV block 数量
        t_gemm   : 每次 GEMM 耗时（相对单位）
        t_softmax: 每次 softmax 耗时
        t_load   : 每次 load 耗时
    """
    # ── FA2：严格串行 ────────────────────────────────────────
    t2_total = 0
    for _ in range(T):
        t2_total += t_load + t_gemm + t_softmax

    # ── FA3：Producer/Consumer pipeline ─────────────────────
    # Producer: load block t
    # Consumer: GEMM(t-1) → phase_A(t-1) → rescale_O → phase_B(t-1)
    # 重叠：load(t) 与 GEMM(t-1) 同时进行
    consumer_time = 0
    producer_time = t_load  # 第一次 load 无重叠

    for t in range(1, T):
        # Consumer 从上一 block 开始工作
        t_consumer_step = t_gemm + t_softmax
        # Producer 同时 load 下一 block
        t_producer_step = t_load
        # 本 step 时间 = max(两者)
        step = max(t_consumer_step, t_producer_step)
        consumer_time += step
        producer_time += step

    # 最后一个 block 的 GEMM+softmax 无重叠
    consumer_time += t_gemm + t_softmax

    t3_total = t_load + consumer_time  # 第一次 load + 流水线

    print(f"\n── Pipeline 重叠模拟（T={T} KV blocks）──")
    print(f"  每 block 耗时：load={t_load:.1f}  GEMM={t_gemm:.1f}  softmax={t_softmax:.1f}")
    print(f"  FA2（串行）   : {t2_total:.1f} 单位时间")
    print(f"  FA3（pipeline): {t3_total:.1f} 单位时间")
    print(f"  理论加速比    : {t2_total / t3_total:.2f}x")


# ──────────────────────────────────────────────────────────────
# 正确性验证
# ──────────────────────────────────────────────────────────────
def verify_fa3_style():
    print("── FA3 两阶段 softmax 正确性验证 ──")
    cases = [
        (1, 1, 128,  64, False, 32, 32),
        (1, 1, 128,  64, True,  32, 32),
        (2, 4, 256, 128, True,  64, 64),
        (1, 1, 200,  64, True,  64, 64),
    ]
    ok = True
    for B, H, N, d, causal, Br, Bc in cases:
        torch.manual_seed(0)
        Q = torch.randn(B, H, N, d)
        K = torch.randn(B, H, N, d)
        V = torch.randn(B, H, N, d)
        scale = 1.0 / math.sqrt(d)

        ref = attention_reference(Q, K, V, scale, causal=causal)
        fa3 = flash_attention_fa3_style(Q, K, V, scale, causal=causal, Br=Br, Bc=Bc)
        err = (ref - fa3).abs().max().item()
        ok &= err < 1e-4
        tag = "causal" if causal else "full  "
        print(f"  [{tag}] N={N:4d} d={d:3d} Br={Br} Bc={Bc}  "
              f"max_err={err:.1e}  {'✓' if err < 1e-4 else '✗'}")
    print(f"  总体：{'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ──────────────────────────────────────────────────────────────
# GPU 性能对比（需要 flash_attn 且有 GPU）
# ──────────────────────────────────────────────────────────────
def gpu_benchmark():
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("\nflash_attn 未安装，跳过 GPU benchmark")
        return
    if not torch.cuda.is_available():
        print("\n无 GPU，跳过 GPU benchmark")
        return

    try:
        from flash_attn_3 import flash_attn_interface as fa3_iface
        has_fa3 = True
    except ImportError:
        has_fa3 = False

    N_list = [512, 1024, 2048, 4096]
    B, H, d = 1, 16, 128
    scale = 1.0 / math.sqrt(d)
    WARMUP, REPEAT = 3, 10

    print("\n── GPU 吞吐量对比（TFLOPS，causal，FP16）──")
    print(f"  {'N':>6}  {'F.sdpa':>10}  {'FA2':>10}  {'FA3':>10}")
    for N in N_list:
        Q = torch.randn(B, N, H, d, dtype=torch.float16, device='cuda')
        K = torch.randn_like(Q); V = torch.randn_like(Q)
        flops = 4 * B * H * N * N * d  # causal ≈ 0.5 * full

        def bench(fn):
            for _ in range(WARMUP): fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEAT): fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / REPEAT

        t_sdpa = bench(lambda: F.scaled_dot_product_attention(
            Q.permute(0,2,1,3), K.permute(0,2,1,3), V.permute(0,2,1,3),
            scale=scale, is_causal=True))
        t_fa2  = bench(lambda: flash_attn_func(Q, K, V, softmax_scale=scale, causal=True))
        tf_fa3 = f"{flops/t_fa3/1e12:.1f}" if has_fa3 else "  N/A"

        if has_fa3:
            t_fa3 = bench(lambda: fa3_iface.flash_attn_func(Q, K, V, causal=True))
            tf_fa3 = f"{flops/t_fa3/1e12:.1f}"

        print(f"  {N:>6}  {flops/t_sdpa/1e12:>10.1f}  "
              f"{flops/t_fa2/1e12:>10.1f}  {tf_fa3:>10}")


if __name__ == "__main__":
    print("=" * 65)
    print(" FA3 Pipeline Simulation & 两阶段 Softmax 验证")
    print("=" * 65)

    verify_fa3_style()
    simulate_pipeline_overlap(T=8, t_gemm=1.0, t_softmax=0.3, t_load=0.8)
    gpu_benchmark()
