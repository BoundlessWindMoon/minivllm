# 03 - Paged KV Cache 内核实现：PagedKVManager 源码精读

> 源码路径：`hopper/paged_kv.h` · `csrc/flash_attn/src/flash_fwd_kernel.h:582-594`

---

## 一、算子定义与公式

### 1.1 Paged KV Cache 的内存模型

标准 KV cache（slot-based）：每条序列独占一块连续的 $L_{\max}$ 大小空间：

$$K_{\text{cache}}[b] \in \mathbb{R}^{L_{\max} \times h_{kv} \times d}, \quad b = 0, \ldots, B-1$$

Paged KV cache：KV 空间切成固定大小的 **page**（每 page $P$ 个 token），物理 page 乱序分配，由 **page table** 记录逻辑→物理映射：

$$\text{page\_table}[b][p] = \phi \in \{0, \ldots, N_{\text{pages}}-1\}$$

物理存储：$K_{\text{phys}}[\phi] \in \mathbb{R}^{P \times h_{kv} \times d}$（$N_{\text{pages}}$ 个物理 page）

逻辑 token $(b, t)$ 的物理地址：

$$\phi = \text{page\_table}[b]\!\left[\left\lfloor t/P \right\rfloor\right]$$

$$\text{offset} = t \bmod P$$

$$K[b, t, :, :] = K_{\text{phys}}\!\left[\phi,\; \text{offset},\; :,\; :\right]$$

### 1.2 Paged Attention 计算

注意力计算公式不变，只是 K/V 的地址需要通过 page table 间接寻址：

$$s_{ij} = q_i \cdot k_{j}^{\text{phys}} \cdot \tau, \quad k_j^{\text{phys}} = K_{\text{phys}}\!\left[\text{page\_table}[b]\!\left[\lfloor j/P \rfloor\right],\; j \bmod P,\; h_k,\; :\right]$$

在 kernel 里，每次加载一个 KV block（$B_c$ 个 token）时：
1. 计算该 block 内所有行对应的 page 编号和页内偏移
2. 查 page table 得物理地址
3. 用 `cp.async` 从各物理地址加载到 SMEM

这个 page table 查询和地址计算完全在 kernel 内完成，Python 层只需传 `block_table`。

### 1.3 内存利用率对比

设序列实际长度服从均匀分布 $\mathcal{U}[L_{\min}, L_{\max}]$：

| 方案 | 每条序列实际占用 | 实际利用率 |
|------|----------------|-----------|
| slot-based（当前） | $L_{\max}$ | $\dfrac{(L_{\min}+L_{\max})/2}{L_{\max}}$ |
| paged（P 个 token/page） | $\lceil L/P \rceil \cdot P$ | $\geq 1 - \dfrac{P-1}{L}$（$L$ 为实际长度） |

当 $L_{\min}=32$，$L_{\max}=4096$，slot-based 利用率约 50%；paged（P=16）利用率 >98%。

### 1.4 FA2 的 page_size 约束

FA2 要求 `page_size`（即 $P$）是 256 的倍数，来自地址对齐推导：

$$\text{k\_batch\_stride} = P \times h_{kv} \times d \times \text{sizeof(Element)}$$

FA2 的 `block_table` 寻址要求 `k_batch_stride` 对 `kBlockN * sizeof(Element) * h_k * d` 对齐，化简后得 $P$ 必须是 256 的倍数。FA3 用 `FastDivmod` 处理任意 $P$，不再有此约束。

---

## 二、为什么 Paged KV 不能用 TMA

TMA（Tensor Memory Accelerator）要求数据在 HBM 中物理连续，通过 descriptor 描述 shape/stride。Paged KV 每个 page 物理地址不连续，TMA descriptor 无法描述间接寻址。

FA3 的 `Use_TMA_KV` 判断（`flash_fwd_kernel_sm90.h:223`）：

```cpp
if constexpr (Use_TMA_KV) {
    pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    // TMA 路径：Producer WG 1 线程触发，物理连续
} else {
    pipeline_params_k.consumer_arv_count = NumMmaThreads;
    // cp.async 路径：Producer WG 全部线程协作 load
}
```

Paged KV 时强制 `Use_TMA_KV=false`，退回 FA2 风格的 `cp.async` 协作加载，但仍保留 Producer/Consumer warp specialization 的框架（Producer WG 做协作式 `cp.async` load，Consumer WG 做 WGMMA）。

---

## 三、PagedKVManager 的核心数据结构

`hopper/paged_kv.h:17`：

```cpp
template <int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads,
          typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
struct PagedKVManager {

    // 128-byte 对齐的 cp.async copy atom（16 个 FP16 = 1 cache line）
    using GmemCopyAtomCpAsync = cute::Copy_Atom<
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;

    // 每行需要几个线程加载？
    // kBlockKGmem = 128（kHeadDim % 128 == 0 时），每线程 16 bytes (8 FP16)
    // kGmemThreadsPerRow = kBlockKGmem / 8 = 16（kHeadDim=128 时）
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;

    // 每个线程负责计算几个 page table entry 的物理地址
    // = ceil(kBlockN 内的行数 / kGmemThreadsPerRow)
    static constexpr int kPageEntryPerThread =
        cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
```

`kPageEntryPerThread` 是关键：每个线程只计算它负责的那些行的物理地址，再通过 `__shfl_sync` 在同一行的线程间广播指针。

---

## 四、load_page_table：逻辑行号 → 物理 page + 页内偏移

```cpp
// paged_kv.h:133-153
template <bool Seqlenk_mask=false, bool First_iter=false>
CUTLASS_DEVICE
void load_page_table(const int n_block) {
    #pragma unroll
    for (int i = 0; i < kPageEntryPerThread; ++i) {
        // 计算这个 thread 负责的第 i 个行的逻辑行号
        // 分配策略：行 0..NumThreads-1 → 线程 0,8,16,...,1,9,17,...（步长 kGmemThreadsPerRow）
        int const row = i * NumThreads
            + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow)
            + (thread_idx / kGmemThreadsPerRow);
        int const row_idx = n_block * kBlockN + row;   // 逻辑 token 位置

        // FastDivmod：用乘法近似除法，避免 20+ cycle 的整数除法延迟
        // page_size_divmod.divmod(quotient, remainder, value)
        int page_idx, page_offset;
        page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
        // leftpad_k：某些 varlen 场景下 KV cache 有起始偏移

        // 查 page table（非合并访问，但每线程只访问自己需要的 entry）
        int const page = (/* OOB check */ ...) ? mPageTable[page_idx] : 0;
        tPrPageOffset[i] = {page, page_offset};  // 存入寄存器
    }
    if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
}
```

**行分配策略的设计考量**（`paged_kv.h:138-143` 注释）：

> "Assuming 8 threads per row, and 176 rows, then the rows from 0 to 175 are loaded by threads 0, 8, 16, ..., 120, 1, 9, ..., 121, 2, 10, ..., 122, etc."

让同一行的 8 个线程（负责加载同一行的不同 128-bit 列片段）各自计算它们需要的 page entry，而不是线程 0 计算全部再广播，减少广播次数。

---

## 五、compute_K_ptr + load_K：物理地址 + cp.async

```cpp
// paged_kv.h:188-233
CUTLASS_DEVICE
TensorKVPtr compute_K_ptr() {
    Tensor tPrKPtr = make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{});
    #pragma unroll
    for (int i = 0; i < kPageEntryPerThread; ++i) {
        auto [page, page_offset] = tPrPageOffset[i];
        // mK_paged(page_offset, 0, page) = K_phys[page][page_offset][head][:]
        tPrKPtr[i] = &mK_paged(page_offset, _0{}, page);
    }
    return tPrKPtr;
}

template <bool Seqlenk_mask=false, typename TensorK>
CUTLASS_DEVICE
void load_K(const int n_block, TensorK &&sK) {
    Tensor tPrKPtr = compute_K_ptr();
    auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});

    for (int m = 0; m < size<1>(tKsK); ++m) {
        bool const should_load = EvenN
            ? (!Seqlenk_mask || ...)
            : get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;

        // __shfl_sync：同一行的 kGmemThreadsPerRow 个线程中，
        // 只有 1 个线程计算了这行的物理指针，广播给其他线程
        Element const* k_ptr = reinterpret_cast<Element const*>(
            __shfl_sync(0xffffffff,
                        reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)),
                        m % kGmemThreadsPerRow,   // src lane
                        kGmemThreadsPerRow));      // width（在小组内广播）

        Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
        // 发起 cp.async，异步加载这行 K 到 SMEM
        cute::copy(GmemCopyAtomCpAsync{}, mK_paged_cur, tKsK(_, m, _));
    }
}
```

`__shfl_sync(mask, var, src_lane, width)` 的 `width=kGmemThreadsPerRow`：只在同行的小组内广播，不需要整个 warp（32 线程）参与，减少 shfl 指令数。

---

## 六、KV_Same_Iter 流水线

`KV_Same_Iter=false`（默认）时，K 的 page table 查询比 V 早一个迭代，流水线如下：

```
iter t-1:  load_page_table(t-1) → compute_K_ptr → cp.async K(t-1)
                                 → compute_V_ptr（预计算 V(t-1) 指针）

iter t  :  load_page_table(t)   → compute_K_ptr → cp.async K(t)
           此时同时 cp.async V(t-1)（指针已在上一轮准备好）

           WGMMA: Q @ K(t-1)^T
           等 V(t-1) 就绪
           WGMMA: P(t-1) @ V(t-1)
```

对应代码（`paged_kv.h:152`）：

```cpp
if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
// 在第一次 load_page_table 时就预计算 V 指针，
// 等下一轮 load_V 时直接用 tPrVPtr，不需要再 compute
```

`get_indices_for_V_TMA`（`paged_kv.h:175-185`）同样实现了 V 比 K 慢一个 iter：

```cpp
CUTLASS_DEVICE
cute::tuple<int, int> get_indices_for_V_TMA() {
    if constexpr (KV_Same_Iter) {
        return {n_block_idx, bidb_kv_idx};
    } else {
        cute::tuple<int, int> const indices = {n_block_idx_prev, bidb_kv_idx_prev};
        bidb_kv_idx_prev = bidb_kv_idx;   // 更新 prev，供下次 V 使用
        n_block_idx_prev = n_block_idx;
        return indices;   // 返回上一轮的 V 索引
    }
}
```

---

## 七、FA2 的 block_table 寻址对比

FA2 在 `compute_attn_1rowblock_splitkv`（`flash_fwd_kernel.h:584-594`）中，每次循环开始时手动计算地址：

```cpp
// FA2：直接 int64_t 算术，无 FastDivmod
const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
const int block_table_idx    = (n_block_max - 1) * kBlockN / params.page_block_size;
const int block_table_offset = (n_block_max - 1) * kBlockN
                               - block_table_idx * params.page_block_size;
const index_t row_offset_k =
    block_table[block_table_idx] * params.k_batch_stride   // 物理 page → base
    + block_table_offset         * params.k_row_stride     // 页内偏移
    + (bidh / h_h_k_ratio)       * params.k_head_stride;   // head 偏移（GQA）
```

FA2 的方案简单但每次迭代都做除法（`/ params.page_block_size`），且 page_size 必须是 256 倍数（使得 `page_block_size` 是 `kBlockN` 倍数，简化 `block_table_idx` 的计算）。FA3 的 `FastDivmod` 去掉了这两个约束。

---

## 八、mini-vllm 改造：从 slot-based 到 paged

当前 `engine/kv_pool.py` 的 `load_kv_for_fa_decode` 返回 slot-indexed 的连续张量：

```python
def load_kv_for_fa_decode(self):
    # 返回 (batch, max_seq_len, nkv, d)，连续内存，可以用 TMA
    return self._pool.k_caches[li][ctx.slot_ids], self._pool.v_caches[li][ctx.slot_ids]
```

改成 paged 后，`k_cache` 是 `(total_pages, page_size, nkv, d)` 的非连续池，传 `block_table`：

```python
# FA2 要求：page_size 是 256 的倍数
PAGE_SIZE = 256  # tokens per page

k_cache = torch.zeros(total_pages, PAGE_SIZE, num_kv_heads, head_dim, ...)
v_cache = torch.zeros_like(k_cache)
block_table = build_block_table(requests, max_pages, device)  # (batch, max_pages)

out = flash_attn_with_kvcache(
    q,                         # (batch, 1, nheads, d)
    k_cache, v_cache,          # (total_pages, PAGE_SIZE, nkv, d)
    k=k_new, v=v_new,
    block_table=block_table,   # (batch, max_pages) int32
    cache_seqlens=lens.int(),  # (batch,) int32
    causal=False,
)
# FA kernel 内部的 PagedKVManager 处理所有地址计算
```

---

## 九、完整可运行代码

代码文件：[`code/paged_kv_sim.py`](code/paged_kv_sim.py)

```
python docs/flash-attention/code/paged_kv_sim.py
```

功能：
1. `PagedKVPool`：完整的物理 page 管理器，实现 §1.1 的 page table 映射
2. `paged_attention`：用 page table 间接寻址做 attention，与 slot-based 结果精确一致
3. 内存利用率对比：不同长度分布下 slot-based vs paged 的实际节省量（典型场景节省 40-50%）
4. FastDivmod 演示：乘法近似除法，PA kernel 在 `load_page_table` 中使用的优化
5. （可选，需 GPU + flash_attn）调用真实 `flash_attn_with_kvcache(block_table=...)` 验证接口
