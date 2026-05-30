"""Qwen3.5 model definition (transformer + causal LM)."""

import torch
from torch import nn
import torch.nn.functional as F

from model.base import BaseCausalLM
from layers.linear import (
    RowParallelLinear,
    QGateKVParallelLinear,
)
from layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from layers.mlp import SiluMLP
from layers.layernorm import RMSNorm, RMSNormGated
from layers.gated_delta_rule import (
    _torch_recurrent_gated_delta_rule,
    _prefill_gated_delta_rule,
)

_FLA_AVAILABLE = False
try:
    from fla.modules.conv import ShortConvolution
    from fla.ops.gated_delta_rule import (
        fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gdn,
    )

    _FLA_AVAILABLE = True
except Exception:
    pass
from layers.rotary_embedding import apply_rotary_pos_emb, RotaryEmbedding
from engine.context import get_context
from utils.model_config import Qwen3_5Config

USE_TRITON = True


class Qwen3_5Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int,
        attention_backend: str = "sdpa",
        use_cuda_graph_bucket: bool = False,
        preallocate_cache: bool = True,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.use_cuda_graph_bucket = use_cuda_graph_bucket
        self.preallocate_cache = preallocate_cache

        self.qkv_gate_proj = QGateKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, centered=True)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, centered=True)

        self.attention_backend = attention_backend
        if attention_backend == "flash_attn":
            try:
                from flash_attn import flash_attn_with_kvcache

                self._flash_attn_with_kvcache = flash_attn_with_kvcache
            except Exception:
                import logging

                logging.warning(
                    "attention_backend='flash_attn' but flash-attn is not installed. "
                    "Falling back to 'sdpa'."
                )
                self.attention_backend = "sdpa"

        self.max_seq_len = getattr(
            config, "kv_cache_max_len", config.max_position_embeddings
        )
        if self.preallocate_cache:
            if self.attention_backend == "flash_attn":
                self.register_buffer(
                    "k_cache",
                    torch.zeros(1, self.max_seq_len, self.num_kv_heads, self.head_dim),
                    persistent=False,
                )
                self.register_buffer(
                    "v_cache",
                    torch.zeros(1, self.max_seq_len, self.num_kv_heads, self.head_dim),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "k_cache",
                    torch.zeros(1, self.num_kv_heads, self.max_seq_len, self.head_dim),
                    persistent=False,
                )
                self.register_buffer(
                    "v_cache",
                    torch.zeros(1, self.num_kv_heads, self.max_seq_len, self.head_dim),
                    persistent=False,
                )
        else:
            self.k_cache = None
            self.v_cache = None
        self.register_buffer(
            "_write_pos",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_attn_mask",
            torch.full((1, 1, 1, self.max_seq_len), float("-inf")),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv_gate = self.qkv_gate_proj(hidden_states)
        q_gate, k, v = qkv_gate.split(
            [
                self.num_heads * self.head_dim * 2,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        query_states, gate = torch.chunk(
            q_gate.view(batch_size, seq_len, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        query_states = (
            self.q_norm(query_states.view(batch_size, seq_len, -1, self.head_dim))
            .transpose(1, 2)
            .contiguous()
        )
        key_states = (
            self.k_norm(k.view(batch_size, seq_len, -1, self.head_dim))
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            v.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2).contiguous()
        )

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        ctx = get_context()
        if ctx:
            is_prefill = ctx.is_prefill
            cache_len = ctx.cache_len
        else:
            is_prefill = True
            cache_len = 0

        if self.attention_backend == "flash_attn":
            q_bshd = query_states.transpose(1, 2)
            k_bshd = key_states.transpose(1, 2)
            v_bshd = value_states.transpose(1, 2)
            if not self.preallocate_cache:
                needed = cache_len + seq_len
                if self.k_cache is None:
                    self.k_cache = torch.zeros(
                        1,
                        needed,
                        self.num_kv_heads,
                        self.head_dim,
                        dtype=k_bshd.dtype,
                        device=k_bshd.device,
                    )
                    self.v_cache = torch.zeros(
                        1,
                        needed,
                        self.num_kv_heads,
                        self.head_dim,
                        dtype=v_bshd.dtype,
                        device=v_bshd.device,
                    )
                elif needed > self.k_cache.shape[1]:
                    pad = needed - self.k_cache.shape[1]
                    self.k_cache = F.pad(self.k_cache, (0, 0, 0, 0, 0, pad, 0, 0))
                    self.v_cache = F.pad(self.v_cache, (0, 0, 0, 0, 0, pad, 0, 0))
            if is_prefill:
                attn_output = self._flash_attn_with_kvcache(
                    q_bshd,
                    self.k_cache,
                    self.v_cache,
                    k=k_bshd,
                    v=v_bshd,
                    cache_seqlens=cache_len,
                    causal=True,
                )
            else:
                attn_output = self._flash_attn_with_kvcache(
                    q_bshd,
                    self.k_cache,
                    self.v_cache,
                    k=k_bshd,
                    v=v_bshd,
                    cache_seqlens=cache_len,
                    causal=False,
                )
            attn_output = attn_output.transpose(1, 2)
        else:
            if is_prefill:
                if not self.preallocate_cache:
                    if self.k_cache is None:
                        self.k_cache = key_states.clone()
                        self.v_cache = value_states.clone()
                    else:
                        needed = cache_len + seq_len
                        if needed > self.k_cache.shape[2]:
                            pad = needed - self.k_cache.shape[2]
                            self.k_cache = F.pad(self.k_cache, (0, 0, 0, pad))
                            self.v_cache = F.pad(self.v_cache, (0, 0, 0, pad))
                        self.k_cache[:, :, cache_len : cache_len + seq_len, :] = (
                            key_states
                        )
                        self.v_cache[:, :, cache_len : cache_len + seq_len, :] = (
                            value_states
                        )
                else:
                    self.k_cache[:, :, cache_len : cache_len + seq_len, :] = key_states
                    self.v_cache[:, :, cache_len : cache_len + seq_len, :] = (
                        value_states
                    )
                k_for_attn = key_states
                v_for_attn = value_states
            else:
                if self.use_cuda_graph_bucket:
                    write_idx = self._write_pos.view(1, 1, 1, 1).expand(
                        batch_size, self.num_kv_heads, seq_len, self.head_dim
                    )
                    self.k_cache.scatter_(2, write_idx, key_states)
                    self.v_cache.scatter_(2, write_idx, value_states)
                    k_for_attn = self.k_cache
                    v_for_attn = self.v_cache
                else:
                    if not self.preallocate_cache:
                        if self.k_cache is None:
                            self.k_cache = key_states.clone()
                            self.v_cache = value_states.clone()
                        else:
                            needed = cache_len + seq_len
                            if needed > self.k_cache.shape[2]:
                                pad = needed - self.k_cache.shape[2]
                                self.k_cache = F.pad(self.k_cache, (0, 0, 0, pad))
                                self.v_cache = F.pad(self.v_cache, (0, 0, 0, pad))
                            self.k_cache[:, :, cache_len : cache_len + seq_len, :] = (
                                key_states
                            )
                            self.v_cache[:, :, cache_len : cache_len + seq_len, :] = (
                                value_states
                            )
                        k_for_attn = self.k_cache
                        v_for_attn = self.v_cache
                    else:
                        self.k_cache[:, :, cache_len : cache_len + seq_len, :] = (
                            key_states
                        )
                        self.v_cache[:, :, cache_len : cache_len + seq_len, :] = (
                            value_states
                        )
                        k_for_attn = self.k_cache[:, :, : cache_len + seq_len, :]
                        v_for_attn = self.v_cache[:, :, : cache_len + seq_len, :]

            if self.num_kv_groups > 1:
                k_for_attn = k_for_attn.repeat_interleave(self.num_kv_groups, dim=1)
                v_for_attn = v_for_attn.repeat_interleave(self.num_kv_groups, dim=1)

            if is_prefill:
                attn_output = F.scaled_dot_product_attention(
                    query_states, k_for_attn, v_for_attn, is_causal=True
                )
            else:
                if self.use_cuda_graph_bucket:
                    attn_output = F.scaled_dot_product_attention(
                        query_states,
                        k_for_attn,
                        v_for_attn,
                        attn_mask=self._attn_mask,
                        is_causal=False,
                    )
                else:
                    attn_output = F.scaled_dot_product_attention(
                        query_states, k_for_attn, v_for_attn, is_causal=False
                    )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# Linear attention (Gated Delta Net)
# ---------------------------------------------------------------------------


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.prefill_backend = getattr(config, "linear_attn_prefill_backend", "torch")
        self.decode_backend = getattr(config, "linear_attn_decode_backend", "fla")

        self.conv_dim = self.key_dim * 2 + self.value_dim
        if self.decode_backend == "fla" and _FLA_AVAILABLE:
            self.conv1d = ShortConvolution(
                hidden_size=self.conv_dim,
                kernel_size=self.conv_kernel_size,
                bias=False,
                activation='silu',
                backend='triton',
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=self.conv_kernel_size,
                groups=self.conv_dim,
                padding=self.conv_kernel_size - 1,
                bias=False,
            )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj = nn.Linear(
            self.hidden_size,
            self.conv_dim + self.value_dim + self.num_v_heads * 2,
            bias=False,
        )

        def _in_proj_weight_loader(param, loaded_weight, shard_id):
            offsets = {
                "qkv": 0,
                "z": self.conv_dim,
                "b": self.conv_dim + self.value_dim,
                "a": self.conv_dim + self.value_dim + self.num_v_heads,
            }
            sizes = {
                "qkv": self.conv_dim,
                "z": self.value_dim,
                "b": self.num_v_heads,
                "a": self.num_v_heads,
            }
            offset = offsets[shard_id]
            size = sizes[shard_id]
            param_data = param.data.narrow(0, offset, size)
            param_data.copy_(loaded_weight)

        self.in_proj.weight.weight_loader = _in_proj_weight_loader

        self.register_buffer(
            "_conv_state",
            torch.zeros(1, self.conv_dim, self.conv_kernel_size),
            persistent=False,
        )
        self.register_buffer(
            "_recurrent_state",
            torch.zeros(1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            persistent=False,
        )
        self._has_state = False

    def reset(self):
        self._has_state = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        ctx = get_context()
        is_prefill = ctx.is_prefill if ctx else True

        proj_out = self.in_proj(hidden_states)
        mixed_qkv, z, b, a = proj_out.split(
            [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads],
            dim=-1,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if self.decode_backend == "fla":
            if not _FLA_AVAILABLE:
                raise RuntimeError("decode_backend='fla' but FLA is not installed")
            # FLA ShortConvolution expects [B, T, D]
            mixed_qkv_t = mixed_qkv.transpose(1, 2)
            mixed_qkv_t, new_conv_state = self.conv1d(
                mixed_qkv_t,
                cache=self._conv_state if self._has_state else None,
                output_final_state=True,
            )
            mixed_qkv = mixed_qkv_t
        else:
            if not is_prefill and self._has_state:
                mixed_qkv = torch.cat(
                    [self._conv_state.to(mixed_qkv.dtype), mixed_qkv], dim=-1
                )
            new_conv_state = F.pad(
                mixed_qkv, (self.conv_kernel_size - 1 - mixed_qkv.shape[-1], 0)
            )
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if not is_prefill and self._has_state:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
            mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if (
            not is_prefill
            and seq_len == 1
            and self._has_state
            and self.decode_backend == "fla"
        ):
            if not _FLA_AVAILABLE:
                raise RuntimeError("decode_backend='fla' but FLA is not installed")
            core_attn_out, last_recurrent_state = _fla_fused_recurrent_gdn(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=self._recurrent_state if self._has_state else None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
            z = z.reshape(-1, self.head_v_dim)
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        elif is_prefill:
            core_attn_out, last_recurrent_state = _prefill_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=self._recurrent_state if self._has_state else None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                backend=self.prefill_backend,
            )
            core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
            z = z.reshape(-1, self.head_v_dim)
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        else:
            core_attn_out, last_recurrent_state = _torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=self._recurrent_state if self._has_state else None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
            z = z.reshape(-1, self.head_v_dim)
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        if new_conv_state is not None:
            if self._conv_state.shape != new_conv_state.shape:
                self._conv_state = torch.zeros_like(new_conv_state)
            self._conv_state.copy_(new_conv_state)
        if last_recurrent_state is not None:
            if self._recurrent_state.shape != last_recurrent_state.shape:
                self._recurrent_state = torch.zeros_like(last_recurrent_state)
            self._recurrent_state.copy_(last_recurrent_state)
        self._has_state = True

        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Qwen3_5MLP(SiluMLP):
    """Backward-compatible alias."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config.hidden_size, config.intermediate_size, bias=False)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                config,
                layer_idx,
                use_cuda_graph_bucket=getattr(config, 'use_cuda_graph_bucket', False),
                attention_backend=getattr(config, 'attention_backend', 'sdpa'),
                preallocate_cache=getattr(config, 'preallocate_cache', True),
            )
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, centered=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, centered=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, cos, sin)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, centered=True)
        rope_params = getattr(config, "rope_parameters", {}) or {}
        mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.rotary_emb = RotaryEmbedding(
            head_size=config.head_dim,
            rotary_dim=int(config.head_dim * config.partial_rotary_factor),
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            mrope_section=mrope_section,
        )

    @property
    def language_model(self):
        return self

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        if position_ids is None:
            positions = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
            cos, sin = self.rotary_emb.get_cos_sin(positions)
        elif position_ids.ndim == 3:
            cos, sin = self.rotary_emb.get_cos_sin_3d(
                position_ids, dtype=hidden_states.dtype
            )
        else:
            cos, sin = self.rotary_emb.get_cos_sin(position_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, cos=cos, sin=sin, residual=residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------------------------------------------------------------------
# Causal LM
# ---------------------------------------------------------------------------


class Qwen3_5ForCausalLM(BaseCausalLM):
    packed_modules_mapping = {
        "q_proj": ("qkv_gate_proj", "q"),
        "k_proj": ("qkv_gate_proj", "k"),
        "v_proj": ("qkv_gate_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "in_proj_qkv": ("in_proj", "qkv"),
        "in_proj_z": ("in_proj", "z"),
        "in_proj_b": ("in_proj", "b"),
        "in_proj_a": ("in_proj", "a"),
    }
    supports_cuda_graph = True

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def reset(self) -> None:
        for layer in self.model.language_model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.reset()
            if hasattr(layer, "self_attn"):
                if layer.self_attn.k_cache is not None:
                    layer.self_attn.k_cache.zero_()
                    layer.self_attn.v_cache.zero_()
                if hasattr(layer.self_attn, "_write_pos"):
                    layer.self_attn._write_pos.zero_()
                if hasattr(layer.self_attn, "_attn_mask"):
                    layer.self_attn._attn_mask.fill_(float("-inf"))

    def iter_attention_modules(self):
        for layer in self.model.language_model.layers:
            if hasattr(layer, "self_attn"):
                yield layer.self_attn

    def _snapshot_cuda_graph_state(self):
        snaps = []
        for layer in self.model.language_model.layers:
            if hasattr(layer, "linear_attn"):
                attn = layer.linear_attn
                snaps.append((attn._conv_state.clone(), attn._recurrent_state.clone()))
            elif hasattr(layer, "self_attn"):
                if layer.self_attn.k_cache is not None:
                    snaps.append(
                        (
                            layer.self_attn.k_cache.clone(),
                            layer.self_attn.v_cache.clone(),
                        )
                    )
                else:
                    snaps.append(None)
            else:
                snaps.append(None)
        return snaps

    def _restore_cuda_graph_state(self, snapshot):
        for layer, snap in zip(self.model.language_model.layers, snapshot):
            if snap is None:
                continue
            if hasattr(layer, "linear_attn"):
                conv_state, recurrent_state = snap
                layer.linear_attn._conv_state.copy_(conv_state)
                layer.linear_attn._recurrent_state.copy_(recurrent_state)
            elif hasattr(layer, "self_attn"):
                k_cache, v_cache = snap
                if layer.self_attn.k_cache is not None:
                    layer.self_attn.k_cache.copy_(k_cache)
                    layer.self_attn.v_cache.copy_(v_cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if positions is None:
            seq_len = input_ids.shape[1]
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.model(input_ids=input_ids, position_ids=positions)
        logits = self.lm_head(hidden_states)
        return logits
