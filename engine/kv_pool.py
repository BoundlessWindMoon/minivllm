"""Paged KV cache pool for multi-sequence inference.

Physical layout (per layer):
    k_caches[l]: (total_pages, page_size, num_kv_heads, head_dim)
    v_caches[l]: same shape

Each request gets a block_table: a list mapping logical page index → physical
page id, mirrored in a GPU tensor (_block_table_gpu) that flash_attn consumes
directly.  No gather of the whole KV cache is needed at decode time.

Contracts:
  - Callers must call ensure_pages(req_id, token_pos) before store_kv so the
    required physical page exists.  BatchedModelRunner does this via
    _ensure_pages_for_prefill (prefill) and _run_decode (decode).
  - store_kv reads ctx.block_tables, which BatchedModelRunner populates from
    block_table_for(slot_ids) before every forward pass.
  - FA2 paged mode requires page_size divisible by 256.
"""

from __future__ import annotations

import math
import torch

from layers.kv_cache import KVCacheBackend
from engine.context import get_context


PAGE_SIZE = 256   # FA2 requirement: page_size % 256 == 0


class PagedKVPool:

    def __init__(
        self,
        num_seqs: int,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype,
        page_size: int = PAGE_SIZE,
    ) -> None:
        if page_size % 256 != 0:
            raise ValueError(f"page_size must be divisible by 256 for FA2, got {page_size}")

        self.num_seqs = num_seqs
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.page_size = page_size

        pages_per_seq = math.ceil(max_seq_len / page_size)
        self.pages_per_seq = pages_per_seq
        real_pages = num_seqs * pages_per_seq
        self.total_pages = real_pages

        # +1 dummy sink page: CUDA graph padding rows write here harmlessly.
        self.dummy_page_id: int = real_pages
        alloc_pages = real_pages + 1

        self.k_caches: list[torch.Tensor] = [
            torch.zeros(alloc_pages, page_size, num_kv_heads, head_dim,
                        dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_caches: list[torch.Tensor] = [
            torch.zeros_like(self.k_caches[i]) for i in range(num_layers)
        ]

        self._free_pages: list[int] = list(range(real_pages))
        self._block_tables: dict[str, list[int]] = {}   # req_id → [phys_page, ...]
        self._req_to_slot: dict[str, int] = {}
        self._slot_to_req: dict[int, str] = {}
        self._free_slots: set[int] = set(range(num_seqs))
        # Pages borrowed from PrefixCache; excluded from _free_pages on free().
        self._shared_pages: dict[str, set[int]] = {}   # req_id → set of shared phys pages

        # GPU block table for flash_attn: (num_seqs, pages_per_seq) int32.
        self._block_table_gpu = torch.full(
            (num_seqs, pages_per_seq), self.dummy_page_id,
            dtype=torch.int32, device=device,
        )

    # ------------------------------------------------------------------
    # Allocation (called by Scheduler)
    # ------------------------------------------------------------------

    def allocate(self, request_id: str, prefix_pages: list[int] | None = None) -> int:
        """Allocate a slot for request_id.

        prefix_pages: physical pages already populated by prefix caching.
            These are pre-filled into the block_table without consuming
            _free_pages, and are NOT returned to _free_pages on free().
        """
        if not self._free_slots:
            raise RuntimeError("PagedKVPool: no free slots")
        slot_id = min(self._free_slots)
        self._free_slots.discard(slot_id)
        self._req_to_slot[request_id] = slot_id
        self._slot_to_req[slot_id] = request_id
        self._block_table_gpu[slot_id].fill_(self.dummy_page_id)

        if prefix_pages:
            self._block_tables[request_id] = list(prefix_pages)
            self._shared_pages[request_id] = set(prefix_pages)
            for logical, phys in enumerate(prefix_pages):
                self._block_table_gpu[slot_id, logical] = phys
        else:
            self._block_tables[request_id] = []
            self._shared_pages[request_id] = set()
        return slot_id

    def free(self, slot_id: int) -> None:
        req_id = self._slot_to_req.pop(slot_id, None)
        if req_id is not None:
            shared = self._shared_pages.pop(req_id, set())
            # Only return exclusively-owned pages to the free pool.
            owned = [p for p in self._block_tables.pop(req_id, [])
                     if p not in shared]
            self._free_pages.extend(owned)
            del self._req_to_slot[req_id]
        self._free_slots.add(slot_id)
        self._block_table_gpu[slot_id].fill_(self.dummy_page_id)

    def reset(self) -> None:
        for req_id in list(self._req_to_slot):
            self.free(self._req_to_slot[req_id])

    def ensure_pages(self, request_id: str, token_pos: int) -> None:
        """Allocate a physical page for request_id at token_pos if not yet done."""
        logical_page = token_pos // self.page_size
        pages = self._block_tables[request_id]
        if logical_page < len(pages):
            return
        if not self._free_pages:
            raise RuntimeError("PagedKVPool: out of physical pages")
        phys = self._free_pages.pop()
        pages.append(phys)
        self._block_table_gpu[self._req_to_slot[request_id], logical_page] = phys

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def num_free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def capacity(self) -> int:
        return self.num_seqs

    def block_table_for(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """Return (bs, pages_per_seq) int32 block table for the given slots."""
        return self._block_table_gpu[slot_ids]

    def pages_for(self, request_id: str) -> list[int]:
        """Return the physical pages allocated for request_id."""
        return list(self._block_tables.get(request_id, []))

    def get_layer_view(self, layer_idx: int) -> "PagedKVLayer":
        return PagedKVLayer(pool=self, layer_idx=layer_idx)


class PagedKVLayer(KVCacheBackend):
    """Per-layer view into a PagedKVPool.

    store_kv and load_kv_for_sdpa are vectorized CUDA operations — no Python
    loops over tokens or batches.  Both require ctx.block_tables to be set
    by the caller (BatchedModelRunner does this before every forward pass).
    """

    def __init__(self, pool: PagedKVPool, layer_idx: int) -> None:
        self._pool = pool
        self._layer_idx = layer_idx

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def store_kv(self, k: torch.Tensor, v: torch.Tensor,
                 cache_len: int, is_prefill: bool) -> None:
        ctx = get_context()
        li = self._layer_idx
        pool = self._pool
        bt = ctx.block_tables          # (bs, pages_per_seq) int32 — must be set
        pool_dtype = pool.k_caches[li].dtype
        page_size = pool.page_size

        if is_prefill:
            # k: (B, kv_h, seq_len, d) → k_bshd: (B, seq_len, kv_h, d)
            # seq_len is the *padded* max_chunk; cu_seqlens_q gives actual lengths.
            offsets = ctx.cache_lens   # (B,) long: starting position per request
            seq_len = k.shape[2]
            B = k.shape[0]
            k_bshd = k.permute(0, 2, 1, 3)
            v_bshd = v.permute(0, 2, 1, 3)

            # Actual token count per request from cu_seqlens_q.
            # actual_lens[b] = cu_seqlens_q[b+1] - cu_seqlens_q[b]
            cu_q = ctx.cu_seqlens_q  # (B+1,) int32
            if cu_q is not None:
                actual_lens = (cu_q[1:B+1] - cu_q[:B]).to(torch.long)  # (B,)
            else:
                actual_lens = torch.full((B,), seq_len, dtype=torch.long,
                                         device=k.device)

            # Build a validity mask: mask[b, t] = True if t < actual_lens[b]
            t_idx = torch.arange(seq_len, device=k.device, dtype=torch.long)
            valid = t_idx.unsqueeze(0) < actual_lens.unsqueeze(1)  # (B, seq)

            # token_pos[b, t] = offsets[b] + t
            token_pos = offsets.to(torch.long).unsqueeze(1) + t_idx.unsqueeze(0)  # (B, seq)
            logical_page = (token_pos // page_size).to(torch.long)
            page_offset  = (token_pos %  page_size).to(torch.long)

            # phys_pages[b, t] = bt[b, logical_page[b, t]]
            phys_pages = bt.to(torch.long)[:B].gather(1, logical_page)  # (B, seq)

            # Apply validity mask: route padding tokens to the dummy sink page.
            dummy = pool.dummy_page_id
            phys_pages = torch.where(valid, phys_pages,
                                     torch.full_like(phys_pages, dummy))

            pool.k_caches[li][phys_pages.reshape(-1), page_offset.reshape(-1)] = (
                k_bshd.reshape(-1, pool.num_kv_heads, pool.head_dim).to(pool_dtype)
            )
            pool.v_caches[li][phys_pages.reshape(-1), page_offset.reshape(-1)] = (
                v_bshd.reshape(-1, pool.num_kv_heads, pool.head_dim).to(pool_dtype)
            )
        else:
            # Decode: one token per request.
            # ctx.cache_lens[i] is the position to write for request i.
            cache_lens_t = ctx.cache_lens
            bs = ctx.slot_ids.shape[0]
            num_real = ctx.num_real_reqs if ctx.num_real_reqs is not None else bs

            cl = cache_lens_t[:num_real].to(torch.long)
            phys_pages = bt[:num_real].gather(
                1, (cl // page_size).unsqueeze(1)
            ).squeeze(1).to(torch.long)
            page_offset = (cl % page_size).to(torch.long)

            k_vec = k.permute(0, 2, 1, 3)[:num_real, 0].to(pool_dtype)
            v_vec = v.permute(0, 2, 1, 3)[:num_real, 0].to(pool_dtype)
            pool.k_caches[li][phys_pages, page_offset] = k_vec
            pool.v_caches[li][phys_pages, page_offset] = v_vec

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def load_kv_for_sdpa(self, total_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct dense (B, kv_h, total_len, d) tensors for SDPA.

        Uses block_table for vectorized gather — no Python token loop.
        """
        ctx = get_context()
        li = self._layer_idx
        pool = self._pool
        bt = ctx.block_tables          # (bs, pages_per_seq) int32
        bs = ctx.slot_ids.shape[0]
        page_size = pool.page_size

        # For each (b, t): logical_page = t // page_size, offset = t % page_size
        t_idx = torch.arange(total_len, device=pool.device, dtype=torch.long)
        logical_page = (t_idx // page_size)  # (total_len,)
        page_offset  = (t_idx %  page_size)  # (total_len,)

        # phys[b, t] = bt[b, logical_page[t]]
        phys = bt[:bs].to(torch.long)[:, logical_page]  # (bs, total_len)

        k_out = pool.k_caches[li][phys, page_offset]   # (bs, total_len, kv_h, d)
        v_out = pool.v_caches[li][phys, page_offset]

        # Return (bs, kv_h, total_len, d) for SDPA
        return k_out.permute(0, 2, 1, 3), v_out.permute(0, 2, 1, 3)

    def load_kv_for_fa_decode(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw physical cache tensors for flash_attn paged decode.

        flash_attn_with_kvcache reads the physical tensors via block_table
        (set in ctx.block_tables); the caller must supply the block_table.
        """
        li = self._layer_idx
        return self._pool.k_caches[li], self._pool.v_caches[li]

    def load_kv(self, total_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.load_kv_for_sdpa(total_len)

    def reset(self) -> None:
        pass

    @property
    def max_seq_len(self) -> int:
        return self._pool.max_seq_len


# Legacy alias — existing imports of KVCachePool keep working.
KVCachePool = PagedKVPool
