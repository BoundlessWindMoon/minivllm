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
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()
