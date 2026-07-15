"""Per-step inference context shared between the runner and attention layers."""

from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cache_len: int = 0

    slot_ids: torch.Tensor | None = None
    cache_lens: torch.Tensor | None = None
    attn_mask: torch.Tensor | None = None

    # Prefill only: varlen attention descriptors.
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # CUDA graph batch decode: number of real requests in the padded batch.
    num_real_reqs: int | None = None

    # Paged KV decode: block_table (bs, pages_per_seq) int32 and
    # seq_lens (bs,) int32 — total KV tokens per sequence including current.
    block_tables: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None

    # (reserved for future use)
    static_k_caches: list[torch.Tensor] | None = None
    static_v_caches: list[torch.Tensor] | None = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill: bool = False,
    cache_len: int = 0,
    slot_ids: torch.Tensor | None = None,
    cache_lens: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    num_real_reqs: int | None = None,
    block_tables: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    static_k_caches: list[torch.Tensor] | None = None,
    static_v_caches: list[torch.Tensor] | None = None,
) -> None:
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cache_len=cache_len,
        slot_ids=slot_ids,
        cache_lens=cache_lens,
        attn_mask=attn_mask,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_real_reqs=num_real_reqs,
        block_tables=block_tables,
        seq_lens=seq_lens,
        static_k_caches=static_k_caches,
        static_v_caches=static_v_caches,
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()
