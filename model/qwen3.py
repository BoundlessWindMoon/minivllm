"""Qwen3 model definition (transformer + causal LM)."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3Config

from model.base import BaseCausalLM

from layers.attention import Attention
from layers.layernorm import RMSNorm
from layers.linear import QKVParallelLinear, RowParallelLinear
from layers.mlp import SiluMLP
from layers.rotary_embedding import get_rope
from layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from typing import (
    List,
    Union,
    Iterable,
    Dict,
    Optional,
)


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        attention_backend: str = "sdpa",
        use_cuda_graph_bucket: bool = False,
        kv_cache_max_len: int | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # NOTE: rope_scaling is ignored because our RotaryEmbedding does not
        # support it; the base frequency is passed explicitly via `base=`.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            max_position=max_position,
            max_seq_len=kv_cache_max_len if kv_cache_max_len is not None else max_position,
            attention_backend=attention_backend,
            use_cuda_graph_bucket=use_cuda_graph_bucket,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, use_cache=use_cache)

        output = self.o_proj(o.flatten(2, -1))
        return output


class Qwen3MLP(SiluMLP):
    """Backward-compatible alias; quantization/awq.py checks isinstance(..., Qwen3MLP)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        assert hidden_act == "silu"
        super().__init__(hidden_size, intermediate_size, bias=False)


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            attention_backend=getattr(config, 'attention_backend', 'sdpa'),
            use_cuda_graph_bucket=getattr(config, 'use_cuda_graph_bucket', False),
            kv_cache_max_len=getattr(config, "kv_cache_max_len", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, use_cache=use_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(BaseCausalLM):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    supports_cuda_graph = True

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(self.config)
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        if self.config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def reset(self) -> None:
        for layer in self.model.layers:
            layer.self_attn.attn.k_cache.zero_()
            layer.self_attn.attn.v_cache.zero_()
            if hasattr(layer.self_attn.attn, "_write_pos"):
                layer.self_attn.attn._write_pos.zero_()
            if hasattr(layer.self_attn.attn, "_attn_mask"):
                layer.self_attn.attn._attn_mask.fill_(float("-inf"))

    def _snapshot_cuda_graph_state(self):
        return [
            (layer.self_attn.attn.k_cache.clone(), layer.self_attn.attn.v_cache.clone())
            for layer in self.model.layers
        ]

    def _restore_cuda_graph_state(self, snapshot):
        for layer, (k_cache, v_cache) in zip(self.model.layers, snapshot):
            layer.self_attn.attn.k_cache.copy_(k_cache)
            layer.self_attn.attn.v_cache.copy_(v_cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if positions is None:
            seq_len = input_ids.shape[1]
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.model(input_ids, positions)
        logits = self.lm_head(hidden_states)
        return logits
