# import torch
# from torch import nn
# from typing import Optional
# from layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
# from layers.activation import SiluAndMul
# from layers.attention import Attention
# from layers.layernorm import RMSNorm
# from layers.rotary_embedding import get_rope
# from layers.embed_head import VocabParallelEmbedding, ParallelLMHead

# def _compute_default_rope_parameters(
#     config,
#     device: Optional["torch.device"] = None,
#     seq_len: Optional[int] = None,
#     **rope_kwargs,
# ) -> tuple["torch.Tensor", float]:
#     """
#     Computes the inverse frequencies according to the original RoPE implementation
#     Args:
#         config ([`~transformers.PretrainedConfig`]):
#             The model configuration.
#         device (`torch.device`):
#             The device to use for initialization of the inverse frequencies.
#         seq_len (`int`, *optional*):
#             The current sequence length. Unused for this type of RoPE.
#         rope_kwargs (`Dict`, *optional*):
#             BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
#     Returns:
#         Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
#         post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
#     """
#     if config is not None and len(rope_kwargs) > 0:
#         raise ValueError(
#             "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
#             f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
#         )
#     if len(rope_kwargs) > 0:
#         base = rope_kwargs["base"]
#         dim = rope_kwargs["dim"]
#     elif config is not None:
#         base = config.rope_theta
#         partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
#         head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         dim = int(head_dim * partial_rotary_factor)

#     attention_factor = 1.0  # Unused in this type of RoPE

#     # Compute the inverse frequencies
#     inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
#     return inv_freq, attention_factor

# # TODO: add rope
# ROPE_INIT_FUNCTIONS = {
#     "default": _compute_default_rope_parameters,
# }

# # TODO: Implement Qwen3MLP
# class Qwen3MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
#         self.gate_up_proj = MergedColumnParallelLinear(
#             self.hidden_size,
#             [self.intermediate_size] * 2,
#             bias=False,
#         )
#         self.down_proj = RowParallelLinear(
#             self.intermediate_size,
#             self.hidden_size,
#             bias=False,
#         )

#         assert config.hidden_act == "silu"
#         self.act_fn = config.hidden_act

#     def forward(self, x):
#         gate_up = self.gate_up_proj(x)
#         x = self.act_fn(gate_up)
#         x = self.down_proj(x)
#         return x
        
    
# # TODO: Implement Qwen3Attention
# class Qwen3Attention(nn.Module):

#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         num_kv_heads: int,
#         max_position: int = 4096 * 32,
#         head_dim: int | None = None,
#         rms_norm_eps: float = 1e-06,
#         qkv_bias: bool = False,
#         rope_theta: float = 10000,
#         rope_scaling: tuple | None = None,
#     ) -> None:
#         super().__init__()
#         tp_size = dist.get_world_size()
#         self.total_num_heads = num_heads
#         assert self.total_num_heads % tp_size == 0
#         self.num_heads = self.total_num_heads // tp_size
#         self.total_num_kv_heads = num_kv_heads
#         assert self.total_num_kv_heads % tp_size == 0
#         self.num_kv_heads = self.total_num_kv_heads // tp_size
#         self.head_dim = head_dim or hidden_size // self.total_num_heads
#         self.q_size = self.num_heads * self.head_dim
#         self.kv_size = self.num_kv_heads * self.head_dim
#         self.scaling = self.head_dim ** -0.5
#         self.qkv_bias = qkv_bias

#         self.qkv_proj = QKVParallelLinear(
#             hidden_size,
#             self.head_dim,
#             self.total_num_heads,
#             self.total_num_kv_heads,
#             bias=qkv_bias,
#         )
#         self.o_proj = RowParallelLinear(
#             self.total_num_heads * self.head_dim,
#             hidden_size,
#             bias=False,
#         )
#         self.rotary_emb = get_rope(
#             self.head_dim,
#             rotary_dim=self.head_dim,
#             max_position=max_position,
#             base=rope_theta,
#             rope_scaling=rope_scaling,
#         )
#         self.attn = Attention(
#             self.num_heads,
#             self.head_dim,
#             self.scaling,
#             self.num_kv_heads,
#         )
#         if not self.qkv_bias:
#             self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
#             self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#     ) -> torch.Tensor:
#         qkv = self.qkv_proj(hidden_states)
#         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
#         q = q.view(-1, self.num_heads, self.head_dim)
#         k = k.view(-1, self.num_kv_heads, self.head_dim)
#         v = v.view(-1, self.num_kv_heads, self.head_dim)
#         if not self.qkv_bias:
#             q = self.q_norm(q)
#             k = self.k_norm(k)
#         q, k = self.rotary_emb(positions, q, k)
#         o = self.attn(q, k, v)
#         output = self.o_proj(o.flatten(1, -1))
#         return output
        

# # TODO: Implement it
# class Qwen3RotaryEmbedding(nn.Module):
#     def __init__(self, config, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

# # TODO: Implement this
# class Qwen3RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Qwen3RMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.eps = eps
        
#     @torch.compile
#     def rms_forward(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         orig_dtype = x.dtype
#         x = x.float()
#         var = x.pow(2).mean(dim=-1, keepdim=True)
#         x.mul_(torch.rsqrt(var + self.eps))
#         x = x.to(orig_dtype).mul_(self.weight)
#         return x

#     @torch.compile
#     def add_rms_forward(
#         self,
#         x: torch.Tensor,
#         residual: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         orig_dtype = x.dtype
#         x = x.float().add_(residual.float())
#         residual = x.to(orig_dtype)
#         var = x.pow(2).mean(dim=-1, keepdim=True)
#         x.mul_(torch.rsqrt(var + self.eps))
#         x = x.to(orig_dtype).mul_(self.weight)
#         return x, residual

#     def forward(
#         self,
#         x: torch.Tensor,
#         residual: torch.Tensor | None = None,
#     ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
#         if residual is None:
#             return self.rms_forward(x)
#         else:
#             return self.add_rms_forward(x, residual)

# # TODO: Implement this
# class Qwen3DecoderLayer(nn.Module):

#     def __init__(
#         self,
#         config,
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.self_attn = Qwen3Attention(
#             hidden_size=config.hidden_size,
#             num_heads=config.num_attention_heads,
#             num_kv_heads=config.num_key_value_heads,
#             max_position=config.max_position_embeddings,
#             rms_norm_eps=config.rms_norm_eps,
#             qkv_bias=getattr(config, 'attention_bias', True),
#             head_dim=getattr(config, 'head_dim', None),
#             rope_theta=getattr(config, "rope_theta", 1000000),
#             rope_scaling=getattr(config, "rope_scaling", None),
#         )
#         self.mlp = Qwen3MLP(
#             hidden_size=config.hidden_size,
#             intermediate_size=config.intermediate_size,
#             hidden_act=config.hidden_act,
#         )
#         self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         residual: torch.Tensor | None,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         if residual is None:
#             hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
#         else:
#             hidden_states, residual = self.input_layernorm(hidden_states, residual)
#         hidden_states = self.self_attn(positions, hidden_states)
#         hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
#         hidden_states = self.mlp(hidden_states)
#         return hidden_states, residual

# # TODO: Implement this
# class Qwen3Model(torch.nn.Module): 
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.layers = nn.ModuleList(
#             [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = Qwen3RotaryEmbedding(config=config)

#         # Initialize weights and apply final processing
#         # self.post_init()
    
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#     ) -> torch.Tensor:
#         hidden_states = self.embed_tokens(input_ids)
#         residual = None
#         for layer in self.layers:
#             hidden_states, residual = layer(positions, hidden_states, residual)
#         hidden_states, _ = self.norm(hidden_states, residual)
#         return hidden_states
    
#     # def post_init(self): 
#     #     pass

# # TODO: fix this
# class Qwen3ForCausalLM(nn.Module):
    
#     packed_modules_mapping = {
#         "q_proj": ("qkv_proj", "q"),
#         "k_proj": ("qkv_proj", "k"),
#         "v_proj": ("qkv_proj", "v"),
#         "gate_proj": ("gate_up_proj", 0),
#         "up_proj": ("gate_up_proj", 1),
#     }
    
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.model = Qwen3Model(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         # self.post_init()
        
#     # def post_init(self): 
#     #     pass
    
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#     ) -> torch.Tensor:
#         return self.model(input_ids, positions)


import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3Config

from layers.activation import SiluAndMul
from layers.attention import Attention
from layers.layernorm import RMSNorm
from layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from layers.rotary_embedding import get_rope
from layers.embed_head import VocabParallelEmbedding, ParallelLMHead

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
        self.scaling = self.head_dim ** -0.5
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
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 显式转换为 4D: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # 修改 flatten 维度，输出形状保持 (batch, seq_len, hidden_size)
        output = self.o_proj(o.flatten(2, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


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
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
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


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(self.config)
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size)
        if self.config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        logits = self.lm_head(hidden_states)
        return logits

