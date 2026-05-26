"""RoPE (Rotary Position Embedding) for Qwen3 / Qwen3.5."""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q/K tensors when cos/sin are pre-computed.

    Args:
        q, k: (..., heads, seq, head_dim)
        cos, sin: (..., seq, rotary_dim)  -- must be broadcastable after unsqueeze
        unsqueeze_dim: where to insert the heads dimension (default 1)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return (
        torch.cat([q_embed.to(q.dtype), q_pass], dim=-1),
        torch.cat([k_embed.to(k.dtype), k_pass], dim=-1),
    )


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "cos_sin_cache",
            self._build_cos_sin_cache(device=None),
            persistent=False,
        )

    def _build_cos_sin_cache(self, device=None) -> torch.Tensor:
        inv_freq = self.inv_freq.to(device=device) if device is not None else self.inv_freq
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)

    def _post_materialize_fixup(self, device):
        # WHY: cos_sin_cache is persistent=False; not in state_dict, never
        # restored by load_state_dict -- rebuild after materialization replaces
        # it with empty_like garbage.
        fresh = self._build_cos_sin_cache(device=device).to(self.cos_sin_cache.dtype)
        with torch.no_grad():
            self.cos_sin_cache.copy_(fresh)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # Partial rotary support (e.g. Qwen3.5 partial_rotary_factor=0.25)
        q_rot = query[..., : self.rotary_dim]
        q_pass = query[..., self.rotary_dim :]
        k_rot = key[..., : self.rotary_dim]
        k_pass = key[..., self.rotary_dim :]

        q_rot = apply_rotary_emb(q_rot, cos, sin)
        k_rot = apply_rotary_emb(k_rot, cos, sin)

        query = torch.cat([q_rot, q_pass], dim=-1)
        key = torch.cat([k_rot, k_pass], dim=-1)
        return query, key

    def get_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tensors compatible with apply_rotary_pos_emb.

        Output shape: (batch_size, seq_len, rotary_dim)
        """
        # cos_sin_cache: (max_position, 1, rotary_dim)
        # first half = cos(θ), second half = sin(θ)
        cos_sin = self.cos_sin_cache[positions]  # (batch, seq, 1, rotary_dim)
        cos_sin = cos_sin.squeeze(2)  # (batch, seq, rotary_dim)
        cos_half, sin_half = cos_sin.chunk(2, dim=-1)  # each (batch, seq, rotary_dim/2)
        # Replicate to full rotary_dim to match transformers-style apply_rotary_pos_emb
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)
        return cos, sin

    def get_cos_sin_3d(
        self, position_ids: torch.Tensor, mrope_section: list[int] | None = None, dtype=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for MRoPE (multimodal 3D positions).

        Args:
            position_ids: (3, batch_size, seq_len) -- temporal/height/width positions
            mrope_section: e.g. [11, 11, 10] for rotary_dim=64; uses self.mrope_section if None
            dtype: output dtype (defaults to float32, matching HF behaviour)

        Returns:
            cos, sin: each (batch_size, seq_len, rotary_dim)
        """
        if mrope_section is None:
            mrope_section = self.mrope_section
        if mrope_section is None:
            # Equal split across 3 dimensions (fallback)
            dim_per_section = self.rotary_dim // 2 // 3
            mrope_section = [dim_per_section, dim_per_section, self.rotary_dim // 2 - 2 * dim_per_section]

        # position_ids: (3, batch, seq)
        # inv_freq: (rotary_dim/2,)
        inv_freq = self.inv_freq.to(position_ids.device)  # (rotary_dim/2,)
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # (3, batch, 1, seq)

        # freqs: (3, batch, seq, rotary_dim/2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

        # Interleave MRoPE: start with temporal, overwrite H/W at specific indices
        freqs_t = freqs[0].clone()  # (batch, seq, rotary_dim/2)
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]

        # Duplicate to full rotary_dim (first half = second half)
        emb = torch.cat((freqs_t, freqs_t), dim=-1)  # (batch, seq, rotary_dim)
        cos = emb.cos()
        sin = emb.sin()
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
