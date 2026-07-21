"""
fa2_tiling.py  —  FlashAttention tiling + online softmax，完整 PyTorch 实现

数学等价于标准 attention，但全程不产生 N×N 的 score 矩阵。

依赖：torch >= 2.0（CPU 即可运行，GPU 可选）
可选：flash_attn（用于 GPU 结果对比）

用法：
    python fa2_tiling.py
"""

import math
import time
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 参考实现：标准 O(N²) attention
# ──────────────────────────────────────────────────────────────
def attention_reference(Q, K, V, scale, causal=False):
    """标准 attention，全量 N×N score 矩阵，用作正确性对照。"""
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B, H, N, N)
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=Q.device), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    return torch.matmul(torch.softmax(S, dim=-1), V)


# ──────────────────────────────────────────────────────────────
# FlashAttention：tiling + online softmax
# ──────────────────────────────────────────────────────────────
def flash_attention_tiled(Q, K, V, scale=None, causal=False, Br=64, Bc=64):
    """
    对应 FA2 compute_attn_1rowblock 的 Python 模拟。

    状态（每行 i 独立，对应寄存器变量）：
        m  : 行最大值，初始 -inf         （对应 row_max）
        l  : exp 累加和，初始 0           （对应 row_sum）
        o  : 输出累积（未归一化），初始 0  （对应 acc_o）

    递推（每个 KV block t）：
        m_new  = max(m, rowmax(S_t))
        alpha  = exp(m - m_new)                    # rescale 系数
        p      = exp(S_t - m_new)                  # 本 block 的 unnorm weights
        l      = alpha * l + rowsum(p)
        o      = alpha * o + p @ V_t               # 未归一化
        m      = m_new

    Epilogue：
        o = o / l                                  # 归一化

    Args:
        Q, K, V : (B, H, N, d) float32
        scale   : softmax scale，默认 1/sqrt(d)
        causal  : 是否使用 causal mask
        Br, Bc  : Q block size / KV block size

    Returns:
        O : (B, H, N, d) float32
    """
    B, H, N, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    O_out = torch.zeros_like(Q)

    for b in range(B):
        for h in range(H):
            q = Q[b, h]  # (N, d)
            k = K[b, h]
            v = V[b, h]

            # 外层循环：Q 的行块（对应 m_block）
            for i0 in range(0, N, Br):
                i1 = min(i0 + Br, N)
                qi = q[i0:i1]           # (br, d)
                br = i1 - i0

                # 寄存器状态初始化
                m = torch.full((br,), float('-inf'), dtype=Q.dtype, device=Q.device)
                l = torch.zeros(br,               dtype=Q.dtype, device=Q.device)
                o = torch.zeros(br, d,            dtype=Q.dtype, device=Q.device)

                # 内层循环：KV 的列块（对应 n_block，从尾到头或头到尾均可）
                for j0 in range(0, N, Bc):
                    j1  = min(j0 + Bc, N)
                    kj  = k[j0:j1]          # (bc, d)
                    vj  = v[j0:j1]          # (bc, d)

                    # ─── QK^T * scale ──────────────────────────
                    s = torch.matmul(qi, kj.T) * scale   # (br, bc)

                    # ─── Causal mask ───────────────────────────
                    if causal:
                        rows = torch.arange(i0, i1, device=Q.device).unsqueeze(1)
                        cols = torch.arange(j0, j1, device=Q.device).unsqueeze(0)
                        s = s.masked_fill(cols > rows, float('-inf'))

                    # ─── Online softmax 递推 ────────────────────
                    m_new = torch.maximum(m, s.max(dim=1).values)   # m^(t)
                    alpha = torch.exp(m - m_new)                     # α^(t)
                    p     = torch.exp(s - m_new.unsqueeze(1))        # unnorm weights

                    l = alpha * l + p.sum(dim=1)                     # l^(t)
                    o = alpha.unsqueeze(1) * o + torch.matmul(p, vj) # O^(t)，未归一化
                    m = m_new

                # ─── Epilogue：归一化 ────────────────────────────
                O_out[b, h, i0:i1] = o / l.unsqueeze(1)

    return O_out


# ──────────────────────────────────────────────────────────────
# 正确性验证
# ──────────────────────────────────────────────────────────────
def verify(B, H, N, d, causal, Br, Bc, atol=1e-4, label=""):
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    scale = 1.0 / math.sqrt(d)

    ref = attention_reference(Q, K, V, scale, causal=causal)
    fa  = flash_attention_tiled(Q, K, V, scale, causal=causal, Br=Br, Bc=Bc)

    max_err  = (ref - fa).abs().max().item()
    mean_err = (ref - fa).abs().mean().item()
    ok = max_err < atol
    tag = "causal" if causal else "full  "
    print(f"  [{tag}] {label:20s} B={B} H={H} N={N:4d} d={d:3d} "
          f"Br={Br} Bc={Bc}  max={max_err:.1e}  mean={mean_err:.1e}  "
          f"{'✓ PASS' if ok else '✗ FAIL'}")
    return ok


# ──────────────────────────────────────────────────────────────
# 内存节省演示
# ──────────────────────────────────────────────────────────────
def show_memory_savings(N=4096, d=128):
    print(f"\n── 内存对比 (N={N}, d={d}, FP16) ──")
    score_mb = N * N * 2 / 1024 / 1024
    tile_kb  = 2 * 64 * d * 2 / 1024   # Q tile + K tile，各 (64 × d) FP16
    lse_kb   = N * 4 / 1024            # LSE 向量
    print(f"  标准 attention score 矩阵 : {score_mb:.1f} MB  （每个 head）")
    print(f"  FA tile (Q+K, Br=Bc=64)  : {tile_kb:.1f} KB  （在 SRAM 内）")
    print(f"  FA 额外存 LSE 向量        : {lse_kb:.2f} KB")
    print(f"  HBM 节省比（理论）        : {score_mb * 1024 / tile_kb:.0f}x")


# ──────────────────────────────────────────────────────────────
# IO complexity 对比演示
# ──────────────────────────────────────────────────────────────
def show_io_complexity():
    print("\n── IO Complexity 对比 ──")
    print("  标准: Θ(N²+Nd) HBM 读写  — score 矩阵必须存到 HBM")
    print("  FA  : Θ(N²d/M) HBM 读写  — tile 在 SRAM 内完成计算")
    print()
    M  = 96 * 1024  # A100 SRAM 约 96 KB per SM
    d  = 128
    Ns = [512, 1024, 2048, 4096, 8192]
    print(f"  {'N':>6}  {'标准(MB)':>10}  {'FA(MB)':>10}  {'节省':>8}")
    for N in Ns:
        std_hbm = (N*N + N*d) * 2 / 1e6         # float16 bytes
        fa_hbm  = N*N*d / M * 2 * 2 / 1e6       # Θ(N²d/M), 2 passes (QKV+O)
        print(f"  {N:>6}  {std_hbm:>10.1f}  {fa_hbm:>10.1f}  {std_hbm/fa_hbm:>7.1f}x")


# ──────────────────────────────────────────────────────────────
# GPU 与真实 flash_attn 对比（可选）
# ──────────────────────────────────────────────────────────────
def compare_with_flash_attn():
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print("\nflash_attn 未安装，跳过 GPU 对比")
        return
    if not torch.cuda.is_available():
        print("\n无 GPU，跳过 flash_attn 对比")
        return

    torch.manual_seed(0)
    B, H, N, d = 2, 4, 512, 128
    scale = 1.0 / math.sqrt(d)
    # flash_attn 用 bshd layout，fp16
    Q = torch.randn(B, N, H, d, dtype=torch.float16, device='cuda')
    K = torch.randn(B, N, H, d, dtype=torch.float16, device='cuda')
    V = torch.randn(B, N, H, d, dtype=torch.float16, device='cuda')

    out_fa = flash_attn_func(Q, K, V, softmax_scale=scale, causal=True)
    # SDPA 在有 FA 的情况下内部也会走 FA kernel
    out_ref = F.scaled_dot_product_attention(
        Q.permute(0,2,1,3), K.permute(0,2,1,3), V.permute(0,2,1,3),
        scale=scale, is_causal=True
    ).permute(0,2,1,3)

    err = (out_fa.float() - out_ref.float()).abs().max().item()
    print(f"\nGPU: flash_attn_func vs F.sdpa  max_err={err:.2e} "
          f"({'✓' if err < 0.01 else '✗'})")  # FP16 下容忍度更宽


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print(" FlashAttention Tiling + Online Softmax — 正确性验证")
    print("=" * 65)

    ok = True
    # 基本 case
    ok &= verify(1, 1, 128,  64, False, 32, 32, label="basic full")
    ok &= verify(1, 1, 128,  64, True,  32, 32, label="basic causal")
    # N 不是 block size 整数倍
    ok &= verify(1, 1, 200,  64, True,  64, 64, label="N%Br != 0")
    ok &= verify(1, 1, 130, 128, True,  64, 64, label="N%Bc != 0")
    # 多 batch/head
    ok &= verify(2, 4, 256, 128, True,  64, 64, label="multi-batch")
    # 大 N
    ok &= verify(1, 1, 512,  64, True,  64, 64, label="N=512")

    print(f"\n总体结果: {'✓ 全部通过' if ok else '✗ 有失败'}")

    show_io_complexity()
    show_memory_savings(N=4096, d=128)
    compare_with_flash_attn()
