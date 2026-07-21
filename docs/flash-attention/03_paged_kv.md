# 03 - Paged KV Cache：从 Slot-Based 到 Page Table

---

## 当前 mini-vllm 的 KV 内存模型

`engine/kv_pool.py` 的方案是 **slot-based**：每条请求独占一个连续的 `max_seq_len` 大小的 KV 空间。

```
KVCachePool:
┌──────────────┐
│  slot 0      │  max_seq_len tokens（不管实际用了多少）
│  slot 1      │
│  slot 2      │
│    ...       │
└──────────────┘

k_caches[layer]: (num_slots, max_seq_len, num_kv_heads, head_dim)
```

**问题**：内存浪费。

假设 `max_seq_len=4096`，batch 里有一条 `cache_len=32` 的短序列，它占了 `4096 - 32 = 4064` 个 token 的空间白白浪费。

vLLM 论文给出的数据：传统静态分配平均浪费 **60-80%** 的 KV 内存（因为无法预知每条序列的实际长度）。

---

## PagedAttention 的思想

借鉴操作系统虚拟内存的 page 机制：

- KV cache 被切成固定大小的 **page**（如每 page = 16 tokens）
- 每条序列通过一个 **page table** 记录自己用了哪些物理 page
- 物理 page 按需分配，序列之间不再需要连续、固定大小的空间

```
逻辑视图（seq_0，64 tokens）：
[page_a][page_b][page_c][page_d]

物理存储（page pool，乱序）：
page_3  ← 存着 seq_0 的 token 0-15
page_17 ← 存着 seq_0 的 token 16-31
page_5  ← 存着 seq_0 的 token 32-47
page_9  ← 存着 seq_0 的 token 48-63

page_table[seq_0] = [3, 17, 5, 9]
```

---

## FA2 的 Paged KV 接口

`flash_attn_with_kvcache` 通过 `block_table` 参数直接支持 paged KV：

```python
# page_size 必须是 256 的倍数（FA 的硬约束）
PAGE_SIZE = 256

# KV 物理存储：(total_pages, page_size, nheads_k, headdim)
k_cache = torch.zeros(total_pages, PAGE_SIZE, num_kv_heads, head_dim)
v_cache = torch.zeros_like(k_cache)

# 每条序列的 page 映射：(batch, max_pages_per_seq)
block_table = torch.zeros(batch_size, max_pages_per_seq, dtype=torch.int32)

out = flash_attn_with_kvcache(
    q,              # (batch, 1, nheads, headdim)  decode
    k_cache,        # (total_pages, PAGE_SIZE, nheads_k, headdim)
    v_cache,
    k=k_new, v=v_new,
    block_table=block_table,
    cache_seqlens=lengths.to(torch.int32),
    causal=False,
)
```

FA kernel 内部（`hopper/paged_kv.h`）的 `PagedKVManager` 在每次加载 KV tile 时：
1. 用当前 tile 的 token 位置计算 page 编号：`page_idx = token_pos // page_size`
2. 从 `block_table` 查出物理 page 地址
3. 用 `cp.async` 异步加载该物理 page

这个间接寻址在 SRAM 里完成，不会额外增加 HBM 访问次数。

---

## mini-vllm 改造路径

### 第一步：建立 Page Pool

```python
PAGE_SIZE = 256  # 必须是 256 的倍数，FA 的约束

class PagedKVPool:
    def __init__(self, total_pages, num_layers, num_kv_heads, head_dim, device, dtype):
        # 物理 KV 存储
        self.k_caches = [
            torch.zeros(total_pages, PAGE_SIZE, num_kv_heads, head_dim,
                        dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_caches = [torch.zeros_like(c) for c in self.k_caches]

        # Page 空闲列表
        self._free_pages: list[int] = list(range(total_pages))

    def alloc_pages(self, n: int) -> list[int]:
        if len(self._free_pages) < n:
            raise RuntimeError("OOM: no free pages")
        pages = self._free_pages[-n:]
        self._free_pages = self._free_pages[:-n]
        return pages

    def free_pages(self, pages: list[int]):
        self._free_pages.extend(pages)
```

### 第二步：Request 维护 page table

```python
@dataclass
class Request:
    ...
    page_table: list[int] = field(default_factory=list)  # 物理 page 列表

    def needs_new_page(self) -> bool:
        token_pos = self.cache_len + 1
        return token_pos % PAGE_SIZE == 0 or not self.page_table
```

### 第三步：调用时构造 block_table tensor

```python
def build_block_table(requests, max_pages_per_seq, device):
    batch = len(requests)
    block_table = torch.zeros(batch, max_pages_per_seq, dtype=torch.int32, device=device)
    for i, req in enumerate(requests):
        for j, page_id in enumerate(req.page_table):
            block_table[i, j] = page_id
    return block_table
```

---

## FA3 的 Paged KV：更激进的优化

FA3 (`hopper/paged_kv.h`) 中 `PagedKVManager` 使用 **TMA (Tensor Memory Accelerator)** 异步加载 page，并且在 page table lookup 时用 `FastDivmod` 避免除法：

```cpp
// paged_kv.h
template <int kBlockN, int kHeadDim, ...>
struct PagedKVManager {
    // 同一 warp 内的线程共享 page table entry，减少 divmod 次数
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    ...
};
```

FA3 中 page 大小没有 FA2 的 256 限制（可以更小），但 H100 上 TMA 的最小粒度有约束，实践中 64-256 tokens/page 是合理范围。

---

## 内存利用率对比

| 方案 | 内存利用率 | 适用场景 |
|------|-----------|---------|
| slot-based（mini-vllm 当前） | ~20-40% | 长度均匀、batch 小 |
| paged KV（page=256） | ~85-95% | 长度多样、高并发 |
| prefix caching + paged | 接近 100% | 有公共前缀的场景（system prompt） |

---

## 注意事项

**FA2 的 page_size 约束**（`flash_attn_interface.py:1551`）：

> `page_block_size must be a multiple of 256`

这是 FA2 的硬约束。FA3 放宽了这个限制，可以在 `hopper/flash_attn_interface.py` 中看到不同的参数。

**Prefix caching 扩展**：paged KV 天然支持前缀共享——多个请求可以指向同一批物理 page（只读共享），这是 vLLM prefix caching 的基础，但需要引用计数管理 page 生命周期。
