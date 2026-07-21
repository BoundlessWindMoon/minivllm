# 02 - FA3 SM90 Kernel 源码精读

> 源码路径：`hopper/flash_fwd_kernel_sm90.h` · `hopper/softmax.h` · `hopper/mask.h`

---

## 一、算子定义与公式

FA3 在数学上与 FA2 完全等价（Attention 公式不变），改变的是**计算调度**，利用 H100/Hopper 的新硬件指令消除流水线气泡。

### 1.1 Attention 公式（同 FA2）

$$O = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

Online softmax 递推（参见 01 文档 §1.2）：

$$m^{(t)} = \max(m^{(t-1)},\; \text{rowmax}(S_t))$$

$$\alpha^{(t)} = \exp\!\left(m^{(t-1)} - m^{(t)}\right)$$

$$l^{(t)} = \alpha^{(t)} l^{(t-1)} + \text{rowsum}\!\left(\exp(S_t - m^{(t)})\right)$$

$$O^{(t)} = \alpha^{(t)} O^{(t-1)} + \exp(S_t - m^{(t)}) \cdot V_t$$

### 1.2 FA3 新增：GEMM-Softmax 重叠调度

FA3 把 §1.2 的递推拆成两个阶段，允许与下一次 GEMM 重叠：

**阶段 A（`max_get_scale`）**：只计算 $m^{(t)}$ 和 $\alpha^{(t)}$：

$$\alpha^{(t)} = \exp\!\left((m^{(t-1)} - m^{(t)}) \cdot \tau \log_2 e\right)$$

立即用 $\alpha^{(t)}$ rescale $O^{(t-1)}$（此时 WGMMA 正在计算 $S_{t+1}$）。

**阶段 B（`online_softmax`）**：等 WGMMA 完成后，计算 $\exp(S_t - m^{(t)})$ 和更新 $l^{(t)}$。

时间线：

```
WGMMA(S_t)          WGMMA(S_{t+1})          WGMMA(S_{t+2})
      ↓ done               ↓ done
   max_get_scale(t)     max_get_scale(t+1)
   rescale O(t-1)        rescale O(t)
        ↓                      ↓
   online_softmax(t)      online_softmax(t+1)
   update O(t)             update O(t+1)
```

### 1.3 AppendKV（`flash_attn_with_kvcache` 的 k=k_new 路径）

将新 K/V 写入 cache 并做 attention，两步合一：

$$K_{\text{cache}}[\text{pos}] \leftarrow k_{\text{new}}$$
$$V_{\text{cache}}[\text{pos}] \leftarrow v_{\text{new}}$$
$$O = \text{Attention}(Q,\; K_{\text{cache}}[:\text{pos}+1],\; V_{\text{cache}}[:\text{pos}+1])$$

这等价于 FA2 的 `flash_attn_with_kvcache(k=k_new, v=v_new, cache_seqlens=pos)`，但在 FA3 里 write 和 attention 在同一 kernel 内完成，减少了一次 HBM 往返。

### 1.4 FP8 Softmax 的 Max_offset 技巧

FP8（E4M3）的值域是 $[-448, 448]$，而 $\exp(x - m) \in (0, 1]$。若直接存 FP8 会大量 underflow。

FA3 在计算 exp 时偏移 $\delta = 8$：

$$p_{ij} = \exp_2\!\left(s_{ij} \cdot \tau \log_2 e - m_i \cdot \tau \log_2 e + \delta\right) = 2^\delta \cdot \exp(s_{ij} \cdot \tau - m_i \cdot \tau)$$

相当于把 $p$ 缩放到 $[0, 2^8] = [0, 256]$，充分利用 FP8 的值域。最后在 `finalize` 里补除以 $2^\delta$：

$$l_i^{\text{corrected}} = l_i^{\text{fp8}} \cdot 2^{-\delta}$$

---

## 二、FA3 对 FA2 的根本改变：Warp Specialization

FA2 所有线程既负责 load 又负责 compute，HBM 延迟直接变成 stall。FA3 的答案：**按角色切分 warp group**。

```cpp
// flash_fwd_kernel_sm90.h:308-360
int warp_group_idx = cutlass::canonical_warp_group_idx();
// 每 4 个 warp (128 threads) = 1 warp group

if (warp_group_idx == 0) {    // Producer WG：专门 load
    mainloop.load(params.mainloop, pipeline_k, pipeline_v, ...);
} else {                       // Consumer WG(s)：专门 MMA
    mainloop.mma(params.mainloop, pipeline_k, pipeline_v, ...);
}
```

---

## 三、Register 配额：Producer 故意少分

H100 每 SM 有 65536 个 32-bit 寄存器，所有 warp 共享。

```cpp
// flash_fwd_kernel_sm90.h:309, 361
if (warp_group_idx == 0) {
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    // Use_TMA_KV=true, 2 Consumer WGs 时：LoadRegisterRequirement = 24
} else {
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
    // MmaRegisterRequirement = 240
}
```

配额计算（1 Producer WG + 2 Consumer WG，各 128 threads）：

```
24  × 128  +  240 × 128  +  240 × 128  =  3072 + 30720 + 30720  =  64512  ≤  65536
```

如果 Producer 也用 240：`240 × 128 × 3 = 92160 > 65536`，SM 只能同时调度 0 个 block，occupancy 归零。Producer 只做 TMA 发起，用 24 个寄存器绰绰有余。

---

## 四、Pipeline：barrier 协调 Producer/Consumer

SMEM 里的 pipeline barrier（`flash_fwd_kernel_sm90.h:105-117`）：

```cpp
struct PipelineStorage {
    alignas(16) BarrierQ          barrier_Q;     // Q tile ready
    alignas(16) ClusterBarrier    barrier_O;     // O tile written
    alignas(16) pipeline_k::SharedStorage pipeline_k;
    alignas(16) pipeline_v::SharedStorage pipeline_v;
    alignas(16) pipeline_vt::SharedStorage pipeline_vt;  // V transposed (FP8)
    alignas(16) pipeline_k_new::SharedStorage pipeline_k_new;  // AppendKV
    alignas(16) pipeline_v_new::SharedStorage pipeline_v_new;
};
```

多 stage 流水线（通常 2-3 stages）：

```
stage 0 SMEM: [K_0][K_1]...    ← Producer 向空闲 stage 写
                    ↑
              Consumer 从就绪 stage 读（WGMMA）
                    ↓
              Consumer release → Producer 复用该 stage
```

Producer 循环（`flash_fwd_kernel_sm90.h:328-358`）：

```cpp
for (auto work_tile_info = scheduler.get_initial_work<IsProducer>(...)
     ...; work_tile_info = scheduler.get_next_work<IsProducer>(...)) {

    mainloop.load(pipeline_k, pipeline_v, pipeline_vt,
                  smem_pipe_write, shared_storage,
                  scheduler_prefetch, seqlen_info, block_coord, work_idx);
}
mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, ...);
```

Consumer 循环（`flash_fwd_kernel_sm90.h:376-451`）：

```cpp
for (auto work_tile_info = scheduler.get_initial_work<false>(...); ...; ) {
    tile_valid = mainloop.mma(pipeline_k, pipeline_v, smem_pipe_read,
                              tOrO, softmax, thread_offset, ...);
    // get_next_work 在 epilogue 之前调用，让下一 tile 尽早调度
    work_tile_info = scheduler.get_next_work<false>(..., work_tile_info);
    if (tile_valid)
        epilogue.store(params.epilogue, tOrO, softmax.row_sum, ...);
}
```

---

## 五、TMA：1 个线程替代 128 个线程的协作加载

FA2 加载 1 个 KV tile（64×128 FP16 = 16 KB）：128 个线程各发 1 条 `cp.async`（16 bytes/次），全部 barrier 同步。

FA3 的 Producer WG 用 TMA：

```cpp
// mainloop.load 内部（概念）
if (warp_idx == 0 && lane_predicate) {  // 只需 1 个线程
    cute::copy(tma_load_K, tKgK(_, n_block), tKsK);  // 异步，硬件 DMA
}
pipeline_k.producer_commit(smem_pipe_write, TmaTransactionBytesK);
```

TMA 特性：
1. **1 线程触发**，其余 127 个 Producer 线程可同时发 V 的 TMA 或发 K_new 的 TMA
2. **TMA descriptor**：在 `Params` 里预先准备好 tensor shape/stride/datatype，触发时无需 CPU 干预
3. **prefetch descriptor**（`flash_fwd_kernel_sm90.h:201-204`）：kernel 最开始由 thread0 发起 descriptor 的 L1 prefetch，后续 TMA 无需等 HBM 读取 descriptor

不可用 TMA 的情况：Paged KV cache（物理非连续），退回 `cp.async`（`Use_TMA_KV=false`）。

---

## 六、FA3 的 Softmax：拆分为两阶段

`hopper/softmax.h` 把 FA2 的 `softmax_rescale_o` 拆成两个方法，对应 §1.2 的两阶段：

**阶段 A：`max_get_scale`（`softmax.h:101-124`）**

```cpp
template<bool Is_first, bool Check_inf=false, typename Tensor0>
__forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(),
                                flash::convert_layout_acc_rowcol(acc_s.layout()));
    TensorT scores_scale;
    if constexpr (Is_first) {
        flash::reduce_max<true>(scores, row_max);
        cute::fill(scores_scale, 1.f);          // 第一次 α=1，无需 rescale
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        cute::copy(row_max, scores_max_prev);
        flash::reduce_max<false>(scores, row_max);
        #pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
            // α^(t) = exp2((m^(t-1) - m^(t)) * τlog₂e)
            scores_scale(mi) = exp2f(
                (scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
            row_sum(mi) *= scores_scale(mi);    // 提前 rescale l，但 O 的 rescale 延后
        }
    }
    return scores_scale;    // 返回给 Consumer，在 WGMMA 期间用于 rescale acc_o
}
```

**阶段 B：`online_softmax`（`softmax.h:126-135`）**

```cpp
template<bool Is_first, bool Check_inf=false, typename Tensor0>
__forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
    Tensor scores = make_tensor(acc_s.data(),
                                flash::convert_layout_acc_rowcol(acc_s.layout()));
    // exp2(s * τlog₂e - m^(t) * τlog₂e)，ffma 优化
    flash::scale_apply_exp2<true, Check_inf, Max_offset>(
        scores, row_max, softmax_scale_log2);
    // 只做 thread-local reduce，warp allreduce 留到 finalize
    flash::reduce_sum<Is_first, /*warp_reduce=*/false>(scores, row_sum);
}
```

**`Max_offset` 的 FP8 技巧**（`softmax.h:66-88`）：

```cpp
// Max_offset=8 时，计算 exp2(x * scale - max_scaled + 8.0)
// 相当于对 p 乘以 2^8 = 256，避免 FP8 underflow
tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled + max_offset);
```

`finalize` 中补除（`softmax.h:147-151`）：

```cpp
if constexpr (Max_offset != 0) {
    static constexpr float sum_scale = 1.f / float(1 << Max_offset); // 1/256
    sum *= sum_scale;
}
```

---

## 七、Mask：thread0 坐标优化 + `__viaddmin_s32`

`hopper/mask.h:65-100` 用 thread0 的列坐标（编译期常量）做比较，消除运行时分支：

```cpp
// mask.h:65-66
int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
int const seqlenk_col_limit  = seqlen_k - n_block * kBlockN - thread_col_offset;

// 用 t0ScS（thread0 的坐标，compile-time 已知）
for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m)
            tSrS_rowcol(m, n) = -INFINITY;  // 整列 mask
    }
}
```

causal mask 中的 `__viaddmin_s32`（`mask.h:97`）：

```cpp
// __viaddmin_s32(a, b, c) = min(a + b, c)，单条 PTX 指令
int const col_limit_right = !Seqlenk_mask
    ? row_idx + causal_row_offset
    : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
```

这是 `vadd.s32` + `min` 的融合指令，比两条指令快约 1 cycle。在每个 KV tile 的 mask 紧密循环里有实际意义。

---

## 八、AppendKV：write cache + attention 同一 kernel

`flash_attn_with_kvcache(k=k_new, v=v_new)` 对应 FA3 的 `AppendKV=true` 路径。

Consumer WG 先写 cache（`flash_fwd_kernel_sm90.h:391-405`）：

```cpp
if constexpr (AppendKV) {
    bool tile_new_valid = mainloop.store_kv_new(
        params.mainloop, pipeline_k_new, pipeline_v_new, smem_pipe_read_new,
        threadIdx.x - MmaThreadOffset, shared_storage, seqlen_info, block_coord);
    if (tile_new_valid) {
        // 确保 Consumer 的写入对 TMA DMA engine 可见（跨 proxy domain 需要显式 fence）
        asm volatile ("fence.proxy.async.global;");
        cutlass::arch::NamedBarrier::arrive(
            NumMmaThreads + NumProducerThreads,
            static_cast<uint32_t>(FwdNamedBarriers::AppendKV));
    }
}
```

写完后 Producer WG 等待这个 barrier，再用 TMA 加载刚写入的 K/V 进行 attention。  
`fence.proxy.async.global` 必须有：CUDA 的 L1/L2 cache 和 TMA DMA engine 属于不同的 proxy domain，没有这条 fence，TMA 看到的可能是 cache 中的旧数据。

---

## 九、FA2 vs FA3 实现对比

| 方面 | FA2 (`flash_fwd_kernel.h`) | FA3 (`flash_fwd_kernel_sm90.h`) |
|------|---------------------------|--------------------------------|
| Load 方式 | 全部线程协作 `cp.async` | Producer WG 发 TMA（1 线程触发） |
| MMA 指令 | `mma.sync`（per-warp，sm80） | `wgmma`（per-warpgroup，更宽） |
| Load/MMA 重叠 | double-buffer `cp.async` | Producer/Consumer 完全异步 |
| Softmax | `softmax_rescale_o`（合并） | `max_get_scale` + `online_softmax`（拆分，与 MMA 重叠） |
| Register 分配 | 所有线程相同 | Producer 24，Consumer 240 |
| Paged KV | 软件 block_table 查表 | `PagedKVManager`（cp.async + FastDivmod） |
| FP8 支持 | 否 | 是（Max_offset=8，q/k/v descale 参数） |
| 吞吐量 (H100 FP16) | ~350 TFLOPS | ~740 TFLOPS |

---

## 十、完整可运行代码

代码文件：[`code/fa3_pipeline_sim.py`](code/fa3_pipeline_sim.py)

```
python docs/flash-attention/code/fa3_pipeline_sim.py
```

功能：
1. 验证 FA3 两阶段 softmax（`max_get_scale` + `online_softmax`）与标准 attention 数值等价
2. 用时间单位模拟 Producer/Consumer pipeline 重叠，量化理论加速比
3. （可选）GPU 对比 F.sdpa / FA2 / FA3 的实际 TFLOPS

