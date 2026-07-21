# 04 - FA3 / H100：Warp Specialization 与 TMA Pipeline

FA3 是专门为 Hopper（H100/H800）架构设计的，核心创新是把 FA2 的"同步串行"变成"异步流水线"。

---

## FA2 的瓶颈

在 FA2 的 CUDA kernel 里，同一个 warp 既负责加载 KV 数据，又负责做矩阵乘法（MMA）：

```
时间轴:
→ 加载 K tile → 做 QK MMA → 加载 V tile → 做 PV MMA → ...
      ↑ 等待 HBM                      ↑ 等待 HBM
      （Stall）                        （Stall）
```

H100 有两个新硬件特性没有被利用：
- **TMA（Tensor Memory Accelerator）**：硬件单元，可以异步地从 HBM 搬数据到 SRAM，不占 CUDA core
- **WGMMA（Warpgroup MMA）**：一次可以做 16 warps（一个 warp group）的矩阵乘，比 A100 的 `mma.sync` 更宽

---

## FA3 的解决方案：Warp Specialization

把 block 内的 warps 分成两类角色：

```
Producer WG（1个 warp group，128 threads）：
  - 专门负责用 TMA 异步加载 K/V 到 SRAM
  - 用 pipeline barrier 通知 Consumer "数据就绪"

Consumer WG（1-2个 warp group）：
  - 专门负责 WGMMA：QK^T 和 PV 矩阵乘
  - 同时执行 softmax 更新（online rescale）
  - 等 pipeline barrier "数据就绪" 信号

时间轴（理想情况）：
Producer: [load K0]  [load V0]  [load K1]  [load V1]  ...
Consumer:       [QK0 MMA] [PV0 MMA] [QK1 MMA] [PV1 MMA] ...
               ↑ overlap ↑
```

这样 HBM 加载和矩阵计算完全重叠（overlap），消除了 stall。

---

## FA3 的 Kernel 代码结构

`hopper/flash_fwd_kernel_sm90.h` 中，kernel 入口按 warp group index 分叉：

```cpp
// flash_fwd_kernel_sm90.h:207-215
int warp_group_idx = cutlass::canonical_warp_group_idx();

if (warp_group_idx == 0) {
    // Producer warp group：发起 TMA 加载
    pipeline_params_k.role = MainloopPipelineK::ThreadCategory::Producer;
} else {
    // Consumer warp group(s)：做 MMA
    pipeline_params_k.role = MainloopPipelineK::ThreadCategory::Consumer;
}
```

Producer 发起 TMA 请求后立即进入"等待 barrier"模式，不占用 CUDA core。Consumer 等数据 ready 才开始 MMA，做完之后标记 barrier 让 Producer 知道可以复用 SRAM。

---

## TMA：Tensor Memory Accelerator

TMA 是 H100 上的硬件 DMA 引擎，专门在 HBM 和 SRAM 之间搬运 tensor block。

主要特性：
- **异步**：发起请求后不阻塞，用 barrier 同步
- **2D/3D 支持**：直接描述 tensor 的 shape 和 stride，硬件负责地址计算
- **只需 1 个线程发起**：省去了 FA2 里所有线程协作 load 的开销

FA3 中用 TMA 加载 Q、K、V（`Use_TMA_Q`, `Use_TMA_KV` 编译选项），但 Paged KV 时 TMA 不可用（不连续内存），退回到 `cp.async`。

---

## 两阶段 GEMM-Softmax 重叠

FA3 还有一个优化：把 softmax（算 `max` 和 `sum`）和 PV GEMM 重叠。

标准流程：
```
QK GEMM → softmax → PV GEMM   （softmax 阻塞 PV GEMM）
```

FA3 流程（利用 WGMMA 的 pingpong）：
```
QK GEMM(tile 0)
         ↓
    softmax(tile 0)
    PV GEMM(tile 0) ← 和下面的 QK GEMM 重叠
QK GEMM(tile 1)
         ↓
    softmax(tile 1) + rescale
    PV GEMM(tile 1)
...
```

这需要 WGMMA 可以异步启动，等上一个 GEMM 的结果被 softmax 消费后再开始下一个 GEMM，通过 CUTLASS 的 pipeline 抽象实现。

---

## 共享内存布局

`flash_fwd_kernel_sm90.h:90-117` 的 `SharedStorage` 展示了 FA3 精心设计的 SRAM 复用：

```cpp
struct SharedStorage {
    struct TensorStorage {
        union {
            struct { typename CollectiveMainloop::TensorStorage mainloop; };
            // smem_o 和 smem_v 共用同一块 SRAM（时间上不重叠）
            typename CollectiveEpilogue::TensorStorage epilogue;
        };
    } tensors;

    struct PipelineStorage {
        alignas(16) BarrierQ  barrier_Q;    // Q 数据就绪信号
        alignas(16) ClusterBarrier barrier_O; // O 输出就绪信号
        alignas(16) pipeline_k::SharedStorage pipeline_k;  // K pipeline barrier
        alignas(16) pipeline_v::SharedStorage pipeline_v;  // V pipeline barrier
        ...
    } pipelines;
};
```

关键设计：**output `O` 和 `V` 的 SRAM 空间重叠**。V 加载完做完 PV MMA 后，就不再需要了，这时 O buffer 可以复用这块空间。节省了宝贵的 SRAM。

---

## FA3 的性能数字

| 精度 | 吞吐量 | 利用率 |
|------|--------|--------|
| FP16 | ~740 TFLOPS | ~75% H100 理论峰值 |
| BF16 | ~740 TFLOPS | ~75% |
| FP8  | ~1.2 PFLOPS | ~60% |

FA2 在 H100 上约 350 TFLOPS（FP16），FA3 约 2x 提升。

---

## 什么时候用 FA3？

| 场景 | 建议 |
|------|------|
| 有 H100/H800 | 安装 FA3（`pip install flash-attn-3`），收益明显 |
| A100 / RTX 系列 | FA2 就够了，FA3 不支持 |
| Paged KV | FA3 支持，但失去 TMA（退回 cp.async），收益减小 |
| FP8 KV quant | FA3 的 FP8 forward pass 可以探索与 KIVI 的结合 |

---

## 和 FA4 的关系

FA4（CuTeDSL 实现）已经在仓库里（`flash_attn/cute/` 目录），用 Python DSL 直接生成 CUDA kernel，同时支持 Hopper 和 Blackwell（B200）。目前还是 experimental，接口和 FA3 基本兼容：

```python
from flash_attn.cute import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```
