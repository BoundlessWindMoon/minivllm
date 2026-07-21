# Flash Attention 源码精读

面向已在写推理引擎（mini-vllm）的读者，逐文件深入 FA2/FA3 的 kernel 实现。

---

## 文档索引

| 文档 | 主题 | 核心源文件 |
|------|------|-----------|
| [01_fa2_forward_kernel.md](01_fa2_forward_kernel.md) | FA2 forward kernel 全流程：tile 循环、online softmax CUDA 实现、epilogue | `flash_fwd_kernel.h`, `softmax.h`, `kernel_traits.h` |
| [02_fa3_sm90_kernel.md](02_fa3_sm90_kernel.md) | FA3 Hopper kernel：Warp Specialization、TMA pipeline、softmax 拆分与 GEMM 重叠 | `flash_fwd_kernel_sm90.h`, `softmax.h`, `mask.h` |
| [03_paged_kv_impl.md](03_paged_kv_impl.md) | PagedKVManager 实现：page table 寻址、cp.async 加载、shfl_sync 广播 | `paged_kv.h`, `flash_fwd_kernel.h:582-594` |

---

## 阅读建议

读 FA2 kernel 时建议同时开着源文件，按调用栈跟踪：

```
compute_attn_1rowblock          (flash_fwd_kernel.h:52)
  ├─ Kernel_traits               (kernel_traits.h:51)
  │   ├─ SmemLayoutQ             (swizzle + tile_to_shape)
  │   └─ TiledMma                (SM80_16x8x16 atom)
  ├─ softmax.softmax_rescale_o   (softmax.h:137)
  │   ├─ reduce_max              (softmax.h:53)
  │   ├─ scale_apply_exp2        (softmax.h:66) ← exp2f + ffma 优化
  │   └─ reduce_sum              (softmax.h:59) ← loop 内只 thread-local
  └─ softmax.normalize_softmax_lse (softmax.h:170)
      └─ quad_allreduce_         (softmax.h:38) ← 最终 warp reduce
```

FA3 kernel 的关键分叉：

```
FlashAttnFwdSm90::operator()    (flash_fwd_kernel_sm90.h:177)
  ├─ warp_group_idx == 0        → Producer WG
  │   ├─ warpgroup_reg_dealloc<24>     (让出寄存器给 Consumer)
  │   └─ mainloop.load(...)            (TMA 异步加载 K/V)
  └─ warp_group_idx != 0        → Consumer WG
      ├─ warpgroup_reg_alloc<240>
      ├─ mainloop.mma(...)             (WGMMA)
      │   └─ softmax.max_get_scale + online_softmax  (与 MMA 重叠)
      └─ epilogue.store(...)
```

---

## 代码路径速查

```
flash-attention/
├── csrc/flash_attn/src/
│   ├── flash_fwd_kernel.h      ← FA2 forward 主 kernel（compute_attn_1rowblock）
│   ├── softmax.h               ← FA2 softmax（softmax_rescale_o）
│   ├── kernel_traits.h         ← FA2 编译期参数：tile size、SMEM layout、MMA atom
│   └── block_info.h            ← varlen 序列长度信息
│
└── hopper/
    ├── flash_fwd_kernel_sm90.h ← FA3 主 kernel（FlashAttnFwdSm90::operator()）
    ├── softmax.h               ← FA3 softmax（max_get_scale + online_softmax，拆分版）
    ├── mask.h                  ← FA3 mask（thread0 坐标优化 + __viaddmin_s32）
    ├── paged_kv.h              ← PagedKVManager（cp.async + FastDivmod + shfl_sync）
    ├── seqlen.h                ← SeqlenInfo 系列（varlen offset 计算）
    └── flash.h                 ← Flash_fwd_params 结构体（所有 kernel 参数）

mini-vllm/
├── layers/attention.py         ← FA 调用入口（flash_attn_with_kvcache，varlen_func）
└── engine/kv_pool.py           ← KVCachePool（slot-based，对应 FA 的 cache_batch_idx 路径）
```
