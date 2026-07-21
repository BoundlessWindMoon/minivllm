# 01 - FA2 Forward Kernel 源码精读

> 源码路径：`csrc/flash_attn/src/flash_fwd_kernel.h` · `softmax.h` · `kernel_traits.h`

---

## 一、算子定义与公式

### 1.1 标准 Attention

给定 $Q \in \mathbb{R}^{N \times d}$，$K \in \mathbb{R}^{N \times d}$，$V \in \mathbb{R}^{N \times d}$，缩放因子 $\tau = 1/\sqrt{d}$：

$$S = QK^T \cdot \tau \in \mathbb{R}^{N \times N}$$

$$P = \text{softmax}(S), \quad P_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}$$

$$O = PV \in \mathbb{R}^{N \times d}$$

对第 $i$ 行展开：

$$o_i = \frac{\sum_j e^{s_{ij}} \cdot v_j}{\sum_j e^{s_{ij}}}$$

数值稳定版本（减去行最大值 $m_i = \max_j s_{ij}$）：

$$o_i = \frac{\sum_j e^{s_{ij} - m_i} \cdot v_j}{\sum_j e^{s_{ij} - m_i}}$$

### 1.2 Online Softmax（分块递推）

将 $K$、$V$ 沿序列维度分成 $T = \lceil N/B_c \rceil$ 个大小为 $B_c$ 的块。  
维护三个行级状态（对每行 $i$ 独立）：

$$m_i^{(0)} = -\infty, \quad l_i^{(0)} = 0, \quad O_i^{(0)} = \mathbf{0}$$

第 $t$ 块（$j \in [(t-1)B_c,\ tB_c)$）的更新：

$$\tilde{s}_{ij} = q_i \cdot k_j \cdot \tau$$

$$m_i^{(t)} = \max\!\left(m_i^{(t-1)},\; \max_{j \in \text{block}_t} \tilde{s}_{ij}\right)$$

$$\alpha_i^{(t)} = \exp\!\left(m_i^{(t-1)} - m_i^{(t)}\right) \quad\text{（rescale 系数）}$$

$$l_i^{(t)} = \alpha_i^{(t)} \cdot l_i^{(t-1)} + \sum_{j \in \text{block}_t} \exp\!\left(\tilde{s}_{ij} - m_i^{(t)}\right)$$

$$O_i^{(t)} = \alpha_i^{(t)} \cdot O_i^{(t-1)} + \sum_{j \in \text{block}_t} \exp\!\left(\tilde{s}_{ij} - m_i^{(t)}\right) \cdot v_j$$

注意 $O_i^{(t)}$ 是**未归一化**的，最后一步归一化：

$$O_i = O_i^{(T)} \;/\; l_i^{(T)}$$

可以证明此结果与标准 softmax 精确等价。

### 1.3 LogSumExp（LSE）

$$\text{LSE}_i = m_i^{(T)} + \log l_i^{(T)}$$

用于 backward recomputation 和 Split-KV 合并。Split-KV 合并公式：

$$\text{LSE}_{\text{combined}} = \log\!\left(e^{\text{LSE}_0} + e^{\text{LSE}_1}\right)$$

$$O_{\text{combined}} = \frac{e^{\text{LSE}_0} O_0 + e^{\text{LSE}_1} O_1}{e^{\text{LSE}_{\text{combined}}}}$$

### 1.4 IO Complexity

| 方案 | HBM 访问量 | 需要 $N^2$ 中间矩阵 |
|------|-----------|-------------------|
| 标准 Attention | $\Theta(N^2 + Nd)$ | 是 |
| Flash Attention | $\Theta(N^2 d / M)$ | 否 |

$M$ 是 SRAM 大小。$M \gg d$ 时（A100: $M \approx 96\text{KB}$，$d=128 \Rightarrow M/d \approx 375$），HBM 访问量下降约 375×。FA 已被证明是该 SRAM 约束下 IO-最优的 attention 算法。

---

## 二、Kernel 分工：一个 block 处理什么

FA2 的核心函数是 `compute_attn_1rowblock`（`flash_fwd_kernel.h:52`）：

```cpp
template<typename Kernel_traits, bool Is_dropout, bool Is_causal,
         bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K,
         bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(
    const Params &params,
    const int bidb,    // batch index
    const int bidh,    // head index
    const int m_block  // 负责 Q 的第 m_block 个行块（kBlockM 行）
)
```

Grid 维度 `(ceil(N/kBlockM), num_heads, batch_size)`——每个 block 独立处理 $B_r$ 行 Q 和全部 KV 列，**block 之间无通信**，所有 $O_i^{(t)}$ 的递推在寄存器里完成。

---

## 三、Kernel Traits：编译期参数系统

`kernel_traits.h:51` 的 `Flash_fwd_kernel_traits` 是 FA2 的核心设计：所有 tile size、SMEM layout、copy atom 在编译期确定，生成无运行时分支的特化 CUDA 代码。

```cpp
// kernel_traits.h:15
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {
    using Element      = elem_type;   // FP16 or BF16
    using ElementAccum = float;       // 累加始终 FP32（对应 O_i^(t) 的精度）
    // sm80 上的 16×8×16 MMA，每次处理 16 行 × 16 列 × 16 K-dim
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
};
```

关键参数（对应公式中的 $B_r$、$B_c$）：
- `kBlockM` → $B_r$，每个 block 处理的 Q 行数，典型值 64 或 128
- `kBlockN` → $B_c$，每次迭代加载的 KV 列数，典型值 64 或 128
- `kNWarps`：每 block 的 warp 数，通常 4（`kBlockM=64`）或 8（`kBlockM=128`）

SMEM layout 使用 swizzle 消除 bank conflict：

```cpp
// kernel_traits.h:70-86
static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
static constexpr int kSwizzle    = kBlockKSmem == 32   ? 2  : 3;

using SmemLayoutAtomQ = decltype(
    composition(Swizzle<kSwizzle, 3, 3>{},
                Layout<Shape<_8, Int<kBlockKSmem>>,
                       Stride<Int<kBlockKSmem>, _1>>{}));
using SmemLayoutQ = decltype(
    tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
```

`Swizzle<3,3,3>`（kHeadDim=128）：每 8 行按位异或列地址的第 3-5 位，使得连续 8 行对应不同 bank，消除 128-byte 对齐引起的 32-bank conflict。

SMEM 大小计算（`kernel_traits.h:107-109`）：

```cpp
static constexpr int kSmemQSize  = size(SmemLayoutQ{})  * sizeof(Element);
static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element); // K+V
static constexpr int kSmemSize   = kSmemQSize + kSmemKVSize;
// 例：kBlockM=64, kBlockN=64, d=128, FP16
// = 64*128*2 + 64*128*2*2 = 16KB + 32KB = 48KB
```

---

## 四、Prologue：Q 和首个 K tile 的预加载

```cpp
// flash_fwd_kernel.h:249-271

// 异步发起 Q 的 HBM → SMEM copy（cp.async，不阻塞）
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
    gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    binfo.actual_seqlen_q - m_block * kBlockM   // 边界 predicate
);

// 从尾部开始，预加载最后一个 K block
int n_block = n_block_max - 1;
FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
    gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    binfo.actual_seqlen_k - n_block * kBlockN
);
cute::cp_async_fence();  // 标记一个 async copy 组的结束
```

**为什么从尾部（`n_block_max - 1`）开始 reverse 迭代？**

`flash_fwd_kernel.h:133`：
> "the last block is the only one that needs masking when we read K and V from global memory. Moreover, iterating in reverse might save us 1 register"

`seqlen_k` 不一定是 `kBlockN` 的倍数，最后一块可能不满，需要 predicate mask 处理越界。倒序后，只有第一次外层迭代（原来的最后一块）需要带 mask 的 copy，后续所有迭代都走 `Is_even_MN=true` 的快路径，消除边界检查。

---

## 五、主循环：掩码迭代与无掩码迭代分离

```cpp
// flash_fwd_kernel.h:297-300
// n_masking_steps 在编译期确定（Is_causal 是模板参数）
constexpr int n_masking_steps = (!Is_causal && !Is_local)
    ? 1
    : ((Is_even_MN && Is_causal)
       ? cute::ceil_div(kBlockM, kBlockN)
       : cute::ceil_div(kBlockM, kBlockN) + 1);

// ① 含 causal mask 的迭代（固定次数，#pragma unroll 完全展开）
#pragma unroll
for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
    // ... QK GEMM + apply_mask + softmax_rescale_o + PV GEMM
}

// ② 无 mask 迭代（剩余 KV blocks，不展开，节省寄存器）
for (; n_block >= n_block_min; --n_block) {
    // ... QK GEMM + softmax_rescale_o + PV GEMM（无 mask 调用）
}
```

causal mask 只影响最后 $\lceil B_r/B_c \rceil$ 个 KV block（Q 行 $i$ 只需 mask 掉 $\text{col} > i$ 的位置，这只发生在 $\text{col} \approx \text{row}$ 的区域）。把它们单独展开，让编译器彻底消除分支。

---

## 六、每次 KV-block 迭代的执行顺序

以无掩码迭代（`flash_fwd_kernel.h:378-428`）为例：

```
① cp_async_wait<0>()      等待上一轮 KV 的 cp.async 完成
   __syncthreads()

② copy(V[n_block] → sV)   异步发起本 block 的 V 加载
   cp_async_fence()

③ gemm(acc_s, Q, K)       QK^T GEMM：acc_s = Q_tile @ K_tile^T * scale
                           结果在寄存器 acc_s (kBlockM × kBlockN, FP32)

④ cp_async_wait<0>()      等待 V 加载完成
   __syncthreads()

⑤ copy(K[n_block-1] → sK) 异步发起下一个 K block 加载
   cp_async_fence()

⑥ softmax_rescale_o(acc_s, acc_o, scale_log2)
                           online softmax：更新 m, l，rescale acc_o，
                           对应公式中的 m^(t), α^(t), l^(t), O^(t)

⑦ rP = fp32→fp16(acc_s)   精度转换：FP32 score → FP16（节省寄存器）
⑧ gemm_rs(acc_o, rP, Vt)  PV GEMM：acc_o += rP @ V_tile
                           对应 O_i^(t) 的累积
```

步骤 ⑤ 和 ⑥⑦⑧ 重叠：下一个 K 的 HBM 加载与当前 tile 的 softmax + PV GEMM 同时进行，这是 FA2 用 `cp.async` 实现的 double-buffer 流水。

---

## 七、Online Softmax 的 CUDA 实现

`csrc/flash_attn/src/softmax.h:137`，`softmax_rescale_o` 对应公式 §1.2 的递推：

```cpp
template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
__forceinline__ __device__ void softmax_rescale_o(
    Tensor0 &acc_s,            // QK^T 分数 (MMA=4, MMA_M, MMA_N)，FP32
    Tensor1 &acc_o,            // 输出累积 (MMA=4, MMA_M, MMA_K)，FP32
    float softmax_scale_log2   // τ * log₂e，预先折进去
) {
    // 把 MMA 寄存器 layout 重解释为逻辑 (nrow, ncol) 矩阵
    Tensor scores = make_tensor(acc_s.data(),
                                convert_layout_acc_rowcol(acc_s.layout()));

    if constexpr (Is_first) {
        // 第一个 KV block：直接初始化 m, l
        reduce_max<zero_init=true>(scores, row_max);          // m^(1)
        scale_apply_exp2(scores, row_max, softmax_scale_log2);// exp2(s*τlog2e - m*τlog2e)
        reduce_sum<zero_init=true>(scores, row_sum);          // l^(1)，thread-local only
    } else {
        // 后续 block：保存旧 max，更新 m，计算 rescale 系数
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);                 // m^(t-1)
        reduce_max<zero_init=false>(scores, row_max);         // m^(t) = max(m^(t-1), ...)

        Tensor acc_o_rowcol = make_tensor(acc_o.data(),
                                          convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
            // α = exp2((m^(t-1) - m^(t)) * τlog2e)，对应公式 §1.2 的 α_i^(t)
            float scores_scale = exp2f(
                (scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
            row_sum(mi) *= scores_scale;                      // l rescale
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
                acc_o_rowcol(mi, ni) *= scores_scale;         // O rescale
        }
        scale_apply_exp2(scores, row_max, softmax_scale_log2);// p = exp(s - m^(t))
        reduce_sum<zero_init=false>(scores, row_sum);         // l 累加
    }
}
```

**`exp2f` 代替 `expf` 的原因**（`softmax.h:78-85`）：

$$\exp(x - m) = \exp_2\!\left(x \cdot \log_2 e - m \cdot \log_2 e\right)$$

把 $\tau \cdot \log_2 e$ 预先算好存为 `softmax_scale_log2`，然后调用 `exp2f(x * scale_log2 - max_scaled)`。编译器将 `x * scale_log2 - max_scaled` 识别为 `ffma`（fused multiply-add），比分开的 `fmul + fsub` 少一条指令，实测约 10-15% 速度差异。

**为什么 loop 内的 `reduce_sum` 不做 warp-level allreduce？**（`softmax.h:163` 注释）：

> "We don't do the reduce across threads here since we don't need to use the row_sum. We do that reduce at the end when we need to normalize the softmax."

loop 内 `row_sum` 只参与乘法（rescale），不需要完整的行累加值。warp-level allreduce 留到 epilogue 的 `normalize_softmax_lse` 才做，节省了每次 KV block 迭代内的 `__shfl_xor_sync` 开销（每次 4 条 shfl 指令）。

**`quad_allreduce_`（`softmax.h:38-44`）**：

```cpp
template<typename Operator>
__device__ void quad_allreduce_(Tensor &dst, Tensor &src, Operator &op) {
    #pragma unroll
    for (int i = 0; i < size(dst); i++)
        dst(i) = Allreduce<4>::run(src(i), op);  // 4 次 __shfl_xor_sync
}
```

`Allreduce<4>` 对应 `SM80_16x8x16` MMA atom 的寄存器分布：一行的结果分散在同一 warp 内连续的 4 个线程中，4 次 xor-reduce（步长 1, 2）即可完成行内归约。

---

## 八、Epilogue：归一化写回

```cpp
// flash_fwd_kernel.h:433-492

// 1. 完成最终归一化（对应 O_i = O_i^(T) / l_i^(T)）
Tensor lse = softmax.normalize_softmax_lse<Is_dropout>(
    acc_o, params.scale_softmax, params.rp_dropout);
// normalize_softmax_lse 内：
//   quad_allreduce_(row_sum)            ← 补上 warp-level reduce
//   acc_o[mi, :] /= row_sum[mi]        ← 归一化
//   lse[mi] = row_max * scale + log(row_sum)  ← LSE 供 backward 用

// 2. FP32 → FP16，通过 SMEM 写回 HBM
Tensor rO = convert_type<Element>(acc_o);       // FP32 → FP16/BF16
Tensor sO = make_tensor(sQ.data(), SmemLayoutO{}); // 复用 sQ 的 SMEM（Q 已不再需要）
cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);   // 寄存器 → SMEM
__syncthreads();
copy<Is_even_MN, Is_even_K, false, false>(
    gmem_tiled_copy_O, tOsO, tOgO, ...);        // SMEM → HBM（128-byte 对齐写）
```

SMEM 复用细节：`sO` 和 `sQ` 指向同一块地址（`sQ.data()`）。Q 在 prologue 加载后进入寄存器（`Is_Q_in_regs=true`）或 SMEM（只读），到 epilogue 时 Q 已经用完，SMEM 可以重新解释为 O 的 write buffer，零额外 SMEM 开销。

---

## 九、Split-KV 路径（`flash_attn_with_kvcache` 底层）

decode 时用 `compute_attn_1rowblock_splitkv`（`flash_fwd_kernel.h:498`），多个 block 并行处理不同段的 KV：

```cpp
const int n_blocks_per_split =
    ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
const int n_block_min = n_split_idx * n_blocks_per_split;
int n_block_max = min(..., (n_split_idx + 1) * n_blocks_per_split);
```

每段独立输出 `oaccum[split]` 和 `lseaccum[split]`，最后 `flash_fwd_combine` kernel 按 §1.3 公式合并。

Paged KV 地址计算（`flash_fwd_kernel.h:584-594`）：

```cpp
const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
// 逻辑 token 位置 → page 编号 + 页内偏移
const int block_table_idx    = (n_block_max - 1) * kBlockN / params.page_block_size;
const int block_table_offset = (n_block_max - 1) * kBlockN
                               - block_table_idx * params.page_block_size;
// 物理地址 = 物理 page base + 页内偏移 + head 偏移
const index_t row_offset_k =
    block_table[block_table_idx] * params.k_batch_stride   // 物理 page id × page stride
    + block_table_offset         * params.k_row_stride     // 页内偏移
    + (bidh / h_h_k_ratio)       * params.k_head_stride;   // head 偏移（GQA）
```

---

## 十、完整可运行代码

代码文件：[`code/fa2_tiling.py`](code/fa2_tiling.py)

```
python docs/flash-attention/code/fa2_tiling.py
```

功能：
1. PyTorch 实现 FA2 tiling + online softmax，验证与标准 attention 数值等价（max_err < 1e-4）
2. 展示各 N 下 HBM 访问量对比（IO complexity 数字）
3. 内存峰值节省量化（score 矩阵 vs SMEM tile）
4. （可选，需 GPU + flash_attn）与真实 FA kernel 输出对比

