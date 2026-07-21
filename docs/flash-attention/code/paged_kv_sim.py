"""
paged_kv_sim.py  —  Paged KV Cache 完整模拟

包含：
1. PagedKVPool：物理 page 池，分配/释放 page
2. paged_attention：用 page table 做间接寻址的 attention
3. 与 slot-based attention 的正确性对比
4. 内存利用率对比
5. （可选）与真实 flash_attn_with_kvcache 的对比

依赖：torch >= 2.0
可选：flash_attn（GPU 结果对比）

用法：
    python paged_kv_sim.py
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────
# Paged KV Pool：管理物理 page 的分配与释放
# ──────────────────────────────────────────────────────────────
class PagedKVPool:
    """
    对应 FA kernel 的 k_cache / v_cache + block_table 结构。

    物理存储：
        k_phys[page_id, offset, head, dim]  (total_pages, PAGE_SIZE, nkv, d)
        v_phys[page_id, offset, head, dim]

    Page table（逻辑→物理映射）：
        block_table[seq_id, logical_page_idx] = physical_page_id
    """
    def __init__(self, total_pages: int, page_size: int,
                 num_kv_heads: int, head_dim: int,
                 dtype=torch.float32, device='cpu'):
        self.total_pages  = total_pages
        self.PAGE_SIZE    = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim

        # 物理 KV 存储
        shape = (total_pages, page_size, num_kv_heads, head_dim)
        self.k_phys = torch.zeros(shape, dtype=dtype, device=device)
        self.v_phys = torch.zeros(shape, dtype=dtype, device=device)

        # 空闲 page 列表
        self._free: List[int] = list(range(total_pages))
        # 每条序列的 page table（逻辑 page idx → 物理 page id）
        self._seq_pages: Dict[str, List[int]] = {}

    def alloc_pages(self, seq_id: str, n_pages: int) -> List[int]:
        """为序列 seq_id 分配 n_pages 个物理 page，返回新分配的 page id 列表。"""
        if len(self._free) < n_pages:
            raise RuntimeError(f"OOM: need {n_pages} pages, have {len(self._free)}")
        pages = self._free[-n_pages:]
        self._free = self._free[:-n_pages]
        self._seq_pages.setdefault(seq_id, []).extend(pages)
        return pages

    def free_seq(self, seq_id: str):
        """释放序列 seq_id 占用的所有 page。"""
        pages = self._seq_pages.pop(seq_id, [])
        self._free.extend(pages)

    def get_block_table(self, seq_id: str) -> List[int]:
        return self._seq_pages.get(seq_id, [])

    def write_kv(self, seq_id: str, token_pos: int,
                 k: torch.Tensor, v: torch.Tensor):
        """
        写入 token_pos 位置的 K/V（公式 §1.1 的物理地址计算）。

        token_pos → page_idx = token_pos // PAGE_SIZE
                    page_offset = token_pos % PAGE_SIZE
        physical_page = block_table[seq_id][page_idx]
        """
        page_idx    = token_pos // self.PAGE_SIZE
        page_offset = token_pos %  self.PAGE_SIZE
        pages = self._seq_pages[seq_id]

        # 如果需要新的 page，先分配
        while len(pages) <= page_idx:
            self.alloc_pages(seq_id, 1)

        phys_page = pages[page_idx]
        self.k_phys[phys_page, page_offset] = k   # (nkv, d)
        self.v_phys[phys_page, page_offset] = v   # (nkv, d)

    def read_kv(self, seq_id: str, seq_len: int):
        """
        读取序列 seq_id 的全部 K/V（按逻辑顺序重组）。
        返回 (seq_len, nkv, d) 的连续张量，用于 attention 计算。
        """
        pages = self._seq_pages[seq_id]
        ks, vs = [], []
        for t in range(seq_len):
            page_idx    = t // self.PAGE_SIZE
            page_offset = t %  self.PAGE_SIZE
            phys_page   = pages[page_idx]
            ks.append(self.k_phys[phys_page, page_offset])
            vs.append(self.v_phys[phys_page, page_offset])
        return torch.stack(ks), torch.stack(vs)   # (seq_len, nkv, d)

    @property
    def free_pages(self):
        return len(self._free)

    def utilization(self, seq_lens: Dict[str, int]) -> float:
        """实际使用的 token 数 / 已分配的 page × PAGE_SIZE。"""
        used_tokens  = sum(seq_lens.values())
        alloc_tokens = sum(
            len(pages) * self.PAGE_SIZE
            for pages in self._seq_pages.values()
        )
        return used_tokens / alloc_tokens if alloc_tokens > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Paged Attention：用 page table 做间接寻址的 attention
# ──────────────────────────────────────────────────────────────
def paged_attention(
    q: torch.Tensor,           # (1, nheads, d)  — 当前 token 的 query
    pool: PagedKVPool,
    seq_id: str,
    seq_len: int,              # 包含当前 token 的总长度
    num_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    用 pool 中 seq_id 的 KV cache 做 attention。

    对应 kernel 内部的 PagedKVManager 流程：
        1. load_page_table：逻辑位置 → 物理 page + offset
        2. compute_K_ptr：得到物理地址
        3. cp.async load：物理地址 → SMEM
        4. GEMM + online softmax
    """
    nkv  = pool.num_kv_heads
    d    = pool.head_dim
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # 从 pool 中读取全部 K/V（Python 层模拟 kernel 内的 page table 寻址）
    k_all, v_all = pool.read_kv(seq_id, seq_len)   # (seq_len, nkv, d)

    # 如果是 GQA，展开 KV heads
    nrep = num_heads // nkv
    if nrep > 1:
        k_all = k_all.repeat_interleave(nrep, dim=1)  # (seq_len, nheads, d)
        v_all = v_all.repeat_interleave(nrep, dim=1)

    # q: (1, nheads, d) → (nheads, 1, d)
    q_t = q.transpose(0, 1)               # (nheads, 1, d)
    k_t = k_all.permute(1, 0, 2)          # (nheads, seq_len, d)
    v_t = v_all.permute(1, 0, 2)          # (nheads, seq_len, d)

    s = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # (nheads, 1, seq_len)
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v_t)              # (nheads, 1, d)
    return o.transpose(0, 1)              # (1, nheads, d)


# ──────────────────────────────────────────────────────────────
# Slot-based attention（参考实现）
# ──────────────────────────────────────────────────────────────
def slot_attention(
    q: torch.Tensor,           # (1, nheads, d)
    k_cache: torch.Tensor,     # (seq_len, nkv, d)
    v_cache: torch.Tensor,
    num_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    nkv = k_cache.shape[1]
    d   = k_cache.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    nrep = num_heads // nkv
    if nrep > 1:
        k_cache = k_cache.repeat_interleave(nrep, dim=1)
        v_cache = v_cache.repeat_interleave(nrep, dim=1)

    q_t = q.transpose(0, 1)
    k_t = k_cache.permute(1, 0, 2)
    v_t = v_cache.permute(1, 0, 2)

    s = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v_t)
    return o.transpose(0, 1)


# ──────────────────────────────────────────────────────────────
# 正确性验证：paged == slot-based
# ──────────────────────────────────────────────────────────────
def verify_paged_vs_slot():
    print("── Paged vs Slot-Based Attention 正确性验证 ──")
    torch.manual_seed(42)

    PAGE_SIZE   = 16   # tokens per page
    total_pages = 64
    num_heads   = 8
    nkv         = 4    # GQA: 2 Q heads per KV head
    head_dim    = 64
    scale       = 1.0 / math.sqrt(head_dim)

    pool = PagedKVPool(total_pages, PAGE_SIZE, nkv, head_dim)

    for seq_len in [1, 15, 16, 32, 47, 64]:
        # 生成随机 K/V
        k_seq = torch.randn(seq_len, nkv, head_dim)
        v_seq = torch.randn(seq_len, nkv, head_dim)
        q     = torch.randn(1, num_heads, head_dim)

        # 写入 paged pool（先确保 seq 存在，再写入）
        pool.free_seq("test")
        pool._seq_pages["test"] = []          # 初始化空 page list
        for t in range(seq_len):
            pool.write_kv("test", t, k_seq[t], v_seq[t])

        # Paged attention
        o_paged = paged_attention(q, pool, "test", seq_len, num_heads, scale)

        # Slot-based reference
        o_slot  = slot_attention(q, k_seq, v_seq, num_heads, scale)

        err = (o_paged - o_slot).abs().max().item()
        ok  = err < 1e-5
        print(f"  seq_len={seq_len:3d}  max_err={err:.1e}  {'✓' if ok else '✗ FAIL'}")

    pool.free_seq("test")


# ──────────────────────────────────────────────────────────────
# 内存利用率对比
# ──────────────────────────────────────────────────────────────
def compare_utilization():
    print("\n── 内存利用率对比（slot-based vs paged）──")
    import random
    random.seed(0)

    L_max    = 4096
    nkv      = 8
    head_dim = 128
    dtype_bytes = 2  # FP16
    kv_per_token = nkv * head_dim * 2 * dtype_bytes  # K+V

    for L_min in [32, 128, 512]:
        # 模拟 100 条序列的实际长度
        seq_lens = [random.randint(L_min, L_max) for _ in range(100)]
        avg_len  = sum(seq_lens) / len(seq_lens)

        # Slot-based：每条序列占 L_max
        slot_bytes  = len(seq_lens) * L_max * kv_per_token
        used_bytes  = sum(seq_lens) * kv_per_token
        slot_util   = used_bytes / slot_bytes

        # Paged（PAGE_SIZE=16）：向上取整到 page 边界
        PAGE_SIZE   = 16
        paged_pages = sum(math.ceil(l / PAGE_SIZE) * PAGE_SIZE for l in seq_lens)
        paged_bytes = paged_pages * kv_per_token
        paged_util  = used_bytes / paged_bytes

        print(f"  L_min={L_min:4d} L_max={L_max}  avg_len={avg_len:.0f}")
        print(f"    Slot-based  : {slot_bytes/1024/1024:.1f} MB  "
              f"利用率 {slot_util:.1%}")
        print(f"    Paged(P=16) : {paged_bytes/1024/1024:.1f} MB  "
              f"利用率 {paged_util:.1%}  "
              f"节省 {(slot_bytes - paged_bytes)/1024/1024:.1f} MB")


# ──────────────────────────────────────────────────────────────
# FastDivmod 效果演示（page table 查询的关键优化）
# ──────────────────────────────────────────────────────────────
def demo_fast_divmod():
    print("\n── FastDivmod：乘法近似除法（避免 20+ cycle 延迟）──")
    import time

    PAGE_SIZE = 16
    N = 10_000_000
    tokens = torch.randint(0, 65536, (N,))

    # 普通除法
    t0 = time.perf_counter()
    page_idx    = tokens // PAGE_SIZE
    page_offset = tokens %  PAGE_SIZE
    t1 = time.perf_counter()

    # FastDivmod 等价（乘以 magic number）：M = ceil(2^32 / PAGE_SIZE)
    # page_idx = floor(token * M / 2^32)
    M = (2**32 + PAGE_SIZE - 1) // PAGE_SIZE
    t2 = time.perf_counter()
    fast_page_idx    = (tokens.long() * M) >> 32
    fast_page_offset = tokens - fast_page_idx * PAGE_SIZE
    t3 = time.perf_counter()

    err = (page_idx.long() - fast_page_idx).abs().max().item()
    print(f"  普通除法    : {(t1-t0)*1000:.2f} ms")
    print(f"  乘法近似    : {(t3-t2)*1000:.2f} ms  误差={err}")
    print(f"  (GPU 上效果更显著：整数除法约 20 cycles，乘法约 4 cycles)")


# ──────────────────────────────────────────────────────────────
# 与真实 flash_attn 对比（需要 GPU + flash_attn）
# ──────────────────────────────────────────────────────────────
def compare_with_flash_attn():
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError:
        print("\nflash_attn 未安装，跳过 GPU 对比")
        return
    if not torch.cuda.is_available():
        print("\n无 GPU，跳过对比")
        return

    print("\n── 与 flash_attn_with_kvcache 对比（Paged KV）──")
    torch.manual_seed(0)

    # FA2 要求 page_size 是 256 的倍数
    PAGE_SIZE  = 256
    total_pages = 32
    B, nheads, nkv, d = 2, 8, 4, 128
    seq_lens   = torch.tensor([512, 384], dtype=torch.int32)
    scale      = 1.0 / math.sqrt(d)

    # k_cache: (total_pages, PAGE_SIZE, nkv, d)
    k_cache = torch.randn(total_pages, PAGE_SIZE, nkv, d,
                          dtype=torch.float16, device='cuda')
    v_cache = torch.randn_like(k_cache)

    # block_table: (B, max_pages_per_seq)
    max_pages = math.ceil(seq_lens.max().item() / PAGE_SIZE)
    block_table = torch.zeros(B, max_pages, dtype=torch.int32, device='cuda')
    for b in range(B):
        n_pages = math.ceil(seq_lens[b].item() / PAGE_SIZE)
        block_table[b, :n_pages] = torch.randperm(total_pages)[:n_pages]

    q = torch.randn(B, 1, nheads, d, dtype=torch.float16, device='cuda')

    out = flash_attn_with_kvcache(
        q, k_cache, v_cache,
        block_table=block_table,
        cache_seqlens=seq_lens.cuda(),
        softmax_scale=scale,
        causal=False,
    )
    print(f"  flash_attn_with_kvcache (paged) output shape: {out.shape}")
    print(f"  block_table shape: {block_table.shape}")
    print(f"  k_cache shape: {k_cache.shape}  (total_pages, PAGE_SIZE, nkv, d)")
    print("  ✓ 调用成功")


if __name__ == "__main__":
    print("=" * 65)
    print(" Paged KV Cache 模拟与验证")
    print("=" * 65)

    verify_paged_vs_slot()
    compare_utilization()
    demo_fast_divmod()
    compare_with_flash_attn()
