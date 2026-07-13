"""Slot-based KV cache pool for multi-sequence inference.

Each concurrent sequence occupies one slot (integer 0..num_slots-1).
K/V tensors are stored in bshd layout (slot, seq, heads, dim) per layer.
"""

from __future__ import annotations

import torch

from layers.kv_cache import KVCacheBackend
from engine.context import get_context


class KVCachePool:

    def __init__(
        self,
        num_slots: int,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.k_caches: list[torch.Tensor] = [
            torch.zeros(num_slots, max_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_caches: list[torch.Tensor] = [
            torch.zeros_like(self.k_caches[i]) for i in range(num_layers)
        ]

        self._free_slots: set[int] = set(range(num_slots))
        self._slot_to_req: dict[int, str] = {}

    def allocate(self, request_id: str) -> int:
        """Assign a free slot to request_id and zero its KV tensors."""
        if not self._free_slots:
            raise RuntimeError("KVCachePool exhausted: no free slots")
        slot_id = self._free_slots.pop()
        for layer_idx in range(self.num_layers):
            self.k_caches[layer_idx][slot_id].zero_()
            self.v_caches[layer_idx][slot_id].zero_()
        self._slot_to_req[slot_id] = request_id
        return slot_id

    def free(self, slot_id: int) -> None:
        """Return slot_id to the free pool. KV data is zeroed on next allocate."""
        self._slot_to_req.pop(slot_id, None)
        self._free_slots.add(slot_id)

    def reset(self) -> None:
        """Return all slots to free without reallocating tensors."""
        self._free_slots = set(range(self.num_slots))
        self._slot_to_req.clear()

    def get_layer_view(self, layer_idx: int) -> "KVCacheLayer":
        """Return a per-layer KVCacheBackend view into this pool."""
        return KVCacheLayer(pool=self, layer_idx=layer_idx)

    def num_free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def capacity(self) -> int:
        return self.num_slots


class KVCacheLayer(KVCacheBackend):
    """Per-layer proxy into KVCachePool."""

    def __init__(self, pool: KVCachePool, layer_idx: int) -> None:
        self._pool = pool
        self._layer_idx = layer_idx

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, cache_len: int, is_prefill: bool) -> None:
        ctx = get_context()
        li = self._layer_idx
        slot_ids = ctx.slot_ids

        if is_prefill:
            offsets = ctx.cache_lens
            seq_len = k.shape[2]
            k_bshd = k.permute(0, 2, 1, 3).contiguous()
            v_bshd = v.permute(0, 2, 1, 3).contiguous()
            for i in range(k.shape[0]):
                slot  = slot_ids[i].item()
                start = int(offsets[i].item()) if offsets is not None else 0
                self._pool.k_caches[li][slot, start:start + seq_len] = k_bshd[i]
                self._pool.v_caches[li][slot, start:start + seq_len] = v_bshd[i]
        else:
            cache_lens_t = ctx.cache_lens
            k_bshd = k.permute(0, 2, 1, 3)
            v_bshd = v.permute(0, 2, 1, 3)
            for i in range(k.shape[0]):
                slot = slot_ids[i].item()
                cl = int(cache_lens_t[i].item()) if cache_lens_t is not None else cache_len
                self._pool.k_caches[li][slot, cl] = k_bshd[i, 0]
                self._pool.v_caches[li][slot, cl] = v_bshd[i, 0]

    def load_kv_for_sdpa(self, total_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K/V in bhsd layout for SDPA attention computation."""
        ctx = get_context()
        li = self._layer_idx
        slot_ids = ctx.slot_ids

        if ctx.is_prefill:
            k = self._pool.k_caches[li][slot_ids][:, :total_len].permute(0, 2, 1, 3)
            v = self._pool.v_caches[li][slot_ids][:, :total_len].permute(0, 2, 1, 3)
        else:
            max_kv = int(ctx.cache_lens.max().item()) + 1 if ctx.cache_lens is not None else total_len
            k = self._pool.k_caches[li][slot_ids][:, :max_kv].permute(0, 2, 1, 3)
            v = self._pool.v_caches[li][slot_ids][:, :max_kv].permute(0, 2, 1, 3)
        return k, v

    def load_kv_for_fa_decode(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full K/V cache in bshd layout for flash_attn_with_kvcache."""
        ctx = get_context()
        li = self._layer_idx
        return self._pool.k_caches[li][ctx.slot_ids], self._pool.v_caches[li][ctx.slot_ids]

    def load_kv(self, total_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility shim delegating to load_kv_for_sdpa."""
        return self.load_kv_for_sdpa(total_len)

    def reset(self) -> None:
        pass

    @property
    def max_seq_len(self) -> int:
        return self._pool.max_seq_len
