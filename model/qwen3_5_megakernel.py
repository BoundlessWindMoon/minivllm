"""Megakernel-backed Qwen3.5 model -- persistent multi-block kernel."""

import torch
from torch import nn

from model.megakernel_weights_qwen3_5 import extract_megakernel_weights_qwen3_5
from kernels.megakernel_cuda import _get_module


class Qwen3_5MegakernelForCausalLM(nn.Module):
    """Drop-in replacement for ``Qwen3_5ForCausalLM``.

    Decode uses fused persistent CUDA kernel (one launch per token).
    Prefill falls back to the base model's PyTorch path.
    """

    supports_cuda_graph = False

    def __init__(self, base_model, max_seq_len: int = 4096, variant: str | None = None):
        super().__init__()
        self.config = base_model.config
        self._max_seq_len = max_seq_len

        cfg = self.config
        hidden_size = cfg.hidden_size
        intermediate_size = cfg.intermediate_size
        num_q_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", hidden_size // num_q_heads)
        num_layers = cfg.num_hidden_layers
        vocab_size = cfg.vocab_size

        # Linear attention dims
        num_k_heads = cfg.linear_num_key_heads
        num_v_heads = cfg.linear_num_value_heads
        head_k_dim = cfg.linear_key_head_dim
        head_v_dim = cfg.linear_value_head_dim
        conv_dim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
        conv_kernel_size = cfg.linear_conv_kernel_dim

        self._q_size = num_q_heads * head_dim
        self._kv_size = num_kv_heads * head_dim
        self._attn_scale = 1.0 / (head_dim ** 0.5)

        # Load CUDA module
        self._mod = _get_module(variant or "qwen3_5_ldg")

        # Extract weights
        w = extract_megakernel_weights_qwen3_5(base_model)
        self._embed_weight = w["embed_weight"]
        self._final_norm_weight = w["final_norm_weight"]
        self._lm_head_weight = w["lm_head_weight"]
        self._layer_weights_bytes = w["layer_weights_bytes"]
        self._stacked = w["stacked"]
        self._layer_types = w["layer_types"]

        # Keep base model for prefill
        self._text_model = base_model.model.language_model
        self._lm_head = base_model.lm_head
        self._visual = getattr(base_model.model, "visual", None)
        self._base_get_image_features = getattr(
            base_model, "_get_image_features", None
        )
        self._base_get_rope_index = getattr(base_model, "_get_rope_index", None)

        del base_model
        torch.cuda.empty_cache()

        dev = self._embed_weight.device

        # Precompute RoPE tables
        rope_params = getattr(cfg, "rope_parameters", {}) or {}
        mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self._mrope_section = mrope_section
        rotary_dim = int(head_dim * cfg.partial_rotary_factor)
        self._cos, self._sin = self._precompute_rope_freqs(
            rotary_dim, max_seq_len, theta=cfg.rope_theta, device=dev
        )

        # Full attention KV cache: [num_full_layers, num_kv_heads, max_seq_len, head_dim]
        num_full_layers = sum(1 for t in self._layer_types if t == "full_attention")
        self._k_cache = torch.zeros(
            num_full_layers, num_kv_heads, max_seq_len, head_dim,
            dtype=torch.bfloat16, device=dev
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Linear attention states
        num_linear_layers = sum(1 for t in self._layer_types if t == "linear_attention")
        self._conv_state = torch.zeros(
            num_linear_layers, conv_dim, conv_kernel_size - 1,
            dtype=torch.float32, device=dev
        )
        self._recurrent_state = torch.zeros(
            num_linear_layers, num_v_heads, head_k_dim, head_v_dim,
            dtype=torch.float32, device=dev
        )

        # Query GPU SM count for persistent kernel grid size
        props = torch.cuda.get_device_properties(dev)
        self._num_blocks = props.multi_processor_count

        # Scratch buffers for persistent kernel
        self._hidden = torch.zeros(hidden_size, dtype=torch.bfloat16, device=dev)
        self._g_act = torch.zeros(hidden_size, dtype=torch.float32, device=dev)
        self._g_res = torch.zeros(hidden_size, dtype=torch.float32, device=dev)
        self._g_q = torch.zeros(max(self._q_size * 2, conv_dim), dtype=torch.float32, device=dev)
        self._g_k = torch.zeros(max(self._kv_size, num_k_heads * head_k_dim), dtype=torch.float32, device=dev)
        self._g_v = torch.zeros(max(self._kv_size, num_v_heads * head_v_dim), dtype=torch.float32, device=dev)
        self._g_z = torch.zeros(num_v_heads * head_v_dim, dtype=torch.float32, device=dev)
        self._g_b = torch.zeros(num_v_heads, dtype=torch.float32, device=dev)
        self._g_a = torch.zeros(num_v_heads, dtype=torch.float32, device=dev)
        self._g_attn = torch.zeros(max(self._q_size, num_v_heads * head_v_dim), dtype=torch.float32, device=dev)
        self._g_mlp = torch.zeros(intermediate_size, dtype=torch.float32, device=dev)
        self._g_norm = torch.zeros(hidden_size, dtype=torch.float32, device=dev)

        # LM head reduction scratch
        lm_num_blocks = getattr(self._mod, "LM_HEAD_NUM_BLOCKS", 1184)
        self._block_max_vals = torch.zeros(lm_num_blocks, dtype=torch.float32, device=dev)
        self._block_max_idxs = torch.zeros(lm_num_blocks, dtype=torch.int32, device=dev)

        # Full logits buffer
        self._logits = torch.zeros(vocab_size, dtype=torch.float32, device=dev)

        # Layer index mapping tensors
        full_idx_list = []
        linear_idx_list = []
        full_idx = 0
        linear_idx = 0
        for t in self._layer_types:
            if t == "full_attention":
                full_idx_list.append(full_idx)
                linear_idx_list.append(-1)
                full_idx += 1
            else:
                full_idx_list.append(-1)
                linear_idx_list.append(linear_idx)
                linear_idx += 1
        self._full_layer_idx = torch.tensor(full_idx_list, dtype=torch.int32, device=dev)
        self._linear_layer_idx = torch.tensor(linear_idx_list, dtype=torch.int32, device=dev)

        # Decode state
        self._position = 0
        self.rope_deltas = None
        self.greedy_fast_path = False

    @staticmethod
    def _precompute_rope_freqs(rotary_dim, max_seq_len, theta=10_000_000.0, device="cuda"):
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
        )
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(torch.bfloat16)
        sin = freqs.sin().to(torch.bfloat16)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin

    @classmethod
    def from_model(cls, model, max_seq_len: int = 4096, variant: str | None = None):
        return cls(model, max_seq_len=max_seq_len, variant=variant)

    def reset(self):
        self._position = 0
        self.rope_deltas = None
        self._k_cache.zero_()
        self._v_cache.zero_()
        self._conv_state.zero_()
        self._recurrent_state.zero_()
        for layer in self._text_model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.reset()

    def _snapshot_cuda_graph_state(self):
        """Snapshot decode state for CUDA Graph capture/restore."""
        return (
            self._k_cache.clone(),
            self._v_cache.clone(),
            self._conv_state.clone(),
            self._recurrent_state.clone(),
        )

    def _restore_cuda_graph_state(self, snapshot):
        """Restore decode state after CUDA Graph capture."""
        k_cache, v_cache, conv_state, recurrent_state = snapshot
        self._k_cache.copy_(k_cache)
        self._v_cache.copy_(v_cache)
        self._conv_state.copy_(conv_state)
        self._recurrent_state.copy_(recurrent_state)

    def _sync_states_from_pytorch(self):
        """Copy attention states from PyTorch layers to CUDA buffers after prefill."""
        for layer_idx, layer_type in enumerate(self._layer_types):
            layer = self._text_model.layers[layer_idx]
            if layer_type == "full_attention":
                full_idx = int(self._full_layer_idx[layer_idx].item())
                attn = layer.self_attn
                cache_len = min(attn.k_cache.shape[2], self._k_cache.shape[2])
                self._k_cache[full_idx].zero_()
                self._v_cache[full_idx].zero_()
                self._k_cache[full_idx, :, :cache_len, :].copy_(attn.k_cache[0, :, :cache_len, :])
                self._v_cache[full_idx, :, :cache_len, :].copy_(attn.v_cache[0, :, :cache_len, :])
            else:
                linear_idx = int(self._linear_layer_idx[layer_idx].item())
                attn = layer.linear_attn
                self._conv_state[linear_idx].copy_(attn._conv_state[0])
                self._recurrent_state[linear_idx].copy_(attn._recurrent_state[0])

    def _run_decode(self, input_token_id: int, position: int, cache_len: int) -> int:
        return self._mod.decode(
            input_token_id,
            position,
            cache_len,
            self._layer_weights_bytes,
            self._embed_weight,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos,
            self._sin,
            self._k_cache,
            self._v_cache,
            self._conv_state,
            self._recurrent_state,
            self._hidden,
            self._g_act,
            self._g_res,
            self._g_q,
            self._g_k,
            self._g_v,
            self._g_z,
            self._g_b,
            self._g_a,
            self._g_attn,
            self._g_mlp,
            self._g_norm,
            self._full_layer_idx,
            self._linear_layer_idx,
            self._num_blocks,
            len(self._layer_types),
            self._max_seq_len,
            self._attn_scale,
        )

    def _run_decode_with_logits(
        self, input_token_id: int, position: int, cache_len: int
    ) -> tuple[int, torch.Tensor]:
        token_id, logits = self._mod.decode_with_logits(
            input_token_id,
            position,
            cache_len,
            self._layer_weights_bytes,
            self._embed_weight,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos,
            self._sin,
            self._k_cache,
            self._v_cache,
            self._conv_state,
            self._recurrent_state,
            self._hidden,
            self._g_act,
            self._g_res,
            self._g_q,
            self._g_k,
            self._g_v,
            self._g_z,
            self._g_b,
            self._g_a,
            self._g_attn,
            self._g_mlp,
            self._g_norm,
            self._full_layer_idx,
            self._linear_layer_idx,
            self._num_blocks,
            len(self._layer_types),
            self._max_seq_len,
            self._attn_scale,
        )
        return token_id, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        decode_position: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise NotImplementedError("Megakernel only supports batch_size=1")

        # Multimodal always falls back to PyTorch
        if pixel_values is not None:
            return self._pytorch_forward(
                input_ids, positions, pixel_values, image_grid_thw, mm_token_type_ids
            )

        # Prefill: use PyTorch for multi-token prompts, then sync states to CUDA
        if seq_len > 1:
            logits = self._pytorch_forward(input_ids, positions)
            self._sync_states_from_pytorch()
            return logits

        # Decode: single token via persistent CUDA kernel
        if decode_position is not None:
            pos = decode_position
        else:
            pos = positions[0, 0].item() if positions is not None else self._position
        cache_len = pos + 1
        token_id = int(input_ids[0, 0].item())

        if self.greedy_fast_path:
            next_token = self._run_decode(token_id, pos, cache_len)
            compact = torch.tensor(
                [[[float(next_token)]]],
                dtype=torch.float32, device=self._hidden.device,
            )
            if decode_position is None:
                self._position = pos + 1
            return compact
        else:
            _, logits = self._run_decode_with_logits(token_id, pos, cache_len)
            if decode_position is None:
                self._position = pos + 1
            return logits.unsqueeze(0).unsqueeze(0)

    def _pytorch_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """PyTorch fallback for prefill."""
        batch_size, seq_len = input_ids.shape
        if pixel_values is not None and self._visual is not None:
            inputs_embeds = self._text_model.embed_tokens(input_ids)
            image_embeds = self._base_get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = input_ids == self.config.image_token_id
            if image_mask.any():
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds
                )
        else:
            inputs_embeds = self._text_model.embed_tokens(input_ids)

        if pixel_values is not None and self._base_get_rope_index is not None:
            if self.rope_deltas is None:
                position_ids, rope_deltas = self._base_get_rope_index(
                    input_ids, image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids
                )
                self.rope_deltas = rope_deltas
            else:
                position_ids = torch.arange(seq_len, device=input_ids.device).view(1, 1, -1).expand(3, batch_size, -1)
                delta = self.rope_deltas[:batch_size].view(1, batch_size, 1)
                position_ids = position_ids + delta
        elif positions is not None and positions.ndim == 3:
            position_ids = positions
        elif positions is not None:
            position_ids = positions
        else:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self._text_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        logits = self._lm_head(hidden_states)
        self._position += seq_len
        return logits
