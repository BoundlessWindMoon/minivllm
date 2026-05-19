"""Megakernel-backed Qwen3 model — all tokens go through the CUDA megakernel.

Prefill is handled by calling the megakernel once per prompt token
(just like mega-qwen's ``TritonDecodeBackend.prefill``).  Decode continues
with the same kernel, re-using the KV cache populated during prefill.
"""

import torch
from torch import nn

from model.megakernel_weights import extract_megakernel_weights
from kernels.megakernel_cuda import _get_module


class Qwen3MegakernelForCausalLM(nn.Module):
    """Drop-in replacement for ``Qwen3ForCausalLM``.

    Every token (prefill + decode) runs through the fused CUDA megakernel.
    The base model is only used as a weight source.
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

        self._q_size = num_q_heads * head_dim
        self._kv_size = num_kv_heads * head_dim
        self._attn_scale = 1.0 / (head_dim**0.5)

        # Compile / load CUDA module
        self._mod = _get_module(variant)

        # Extract and prepare weights
        w = extract_megakernel_weights(base_model)
        self._embed_weight = w["embed_weight"]
        self._final_norm_weight = w["final_norm_weight"]
        self._lm_head_weight = w["lm_head_weight"]
        self._layer_weights_bytes = w["layer_weights_bytes"]
        self._stacked = w["stacked"]  # keep references alive

        # Original model weights are no longer needed; free them.
        del base_model
        torch.cuda.empty_cache()

        # Query GPU SM count for persistent kernel grid size
        props = torch.cuda.get_device_properties(self._embed_weight.device)
        self._num_blocks = props.multi_processor_count

        # Precompute RoPE tables
        self._cos, self._sin = self._precompute_rope_freqs(
            head_dim,
            max_seq_len,
            theta=getattr(cfg, "rope_theta", 1_000_000.0),
            device=self._embed_weight.device,
        )

        # KV cache: [num_layers, num_kv_heads, max_seq_len, head_dim] bf16
        self._k_cache = torch.zeros(
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=torch.bfloat16,
            device=self._embed_weight.device,
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
        dev = self._embed_weight.device
        self._hidden = torch.zeros(hidden_size, dtype=torch.bfloat16, device=dev)
        self._g_act = torch.zeros(hidden_size, dtype=torch.float32, device=dev)
        self._g_res = torch.zeros(hidden_size, dtype=torch.float32, device=dev)
        self._g_q = torch.zeros(self._q_size, dtype=torch.float32, device=dev)
        self._g_k = torch.zeros(self._kv_size, dtype=torch.float32, device=dev)
        self._g_v = torch.zeros(self._kv_size, dtype=torch.float32, device=dev)
        self._g_attn = torch.zeros(self._q_size, dtype=torch.float32, device=dev)
        self._g_mlp = torch.zeros(intermediate_size, dtype=torch.float32, device=dev)
        self._g_norm = torch.zeros(hidden_size, dtype=torch.float32, device=dev)

        # LM head reduction scratch – size comes from kernel constants
        lm_num_blocks = getattr(self._mod, "LM_HEAD_NUM_BLOCKS", 1184)
        self._block_max_vals = torch.zeros(
            lm_num_blocks, dtype=torch.float32, device=dev
        )
        self._block_max_idxs = torch.zeros(lm_num_blocks, dtype=torch.int32, device=dev)

        # Full logits buffer (for decode_with_logits)
        self._logits = torch.zeros(vocab_size, dtype=torch.float32, device=dev)

        # Greedy fast-path flag (set by main.py based on sampling config)
        self.greedy_fast_path = False

        # Decode state
        self._position = 0

    @staticmethod
    def _precompute_rope_freqs(head_dim, max_seq_len, theta=1_000_000.0, device="cuda"):
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
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
        """Factory: wrap an already-loaded mini-vllm model."""
        return cls(model, max_seq_len=max_seq_len, variant=variant)

    def reset(self):
        """Reset decode state and KV cache."""
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

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
            self._hidden,
            self._g_act,
            self._g_res,
            self._g_q,
            self._g_k,
            self._g_v,
            self._g_attn,
            self._g_mlp,
            self._g_norm,
            self._block_max_vals,
            self._block_max_idxs,
            self._num_blocks,
            self.config.num_hidden_layers,
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
            self._hidden,
            self._g_act,
            self._g_res,
            self._g_q,
            self._g_k,
            self._g_v,
            self._g_attn,
            self._g_mlp,
            self._g_norm,
            self._block_max_vals,
            self._block_max_idxs,
            self._num_blocks,
            self.config.num_hidden_layers,
            self._max_seq_len,
            self._attn_scale,
        )
        return token_id, logits

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None):
        """Model forward.

        Returns logits of shape ``[batch, seq_len, vocab_size]``.

        All tokens go through the megakernel.  For prefill (seq_len > 1) we
        loop over prompt tokens one-by-one, just like mega-qwen does.
        """
        batch_size, seq_len = input_ids.shape

        if batch_size != 1:
            raise NotImplementedError("Megakernel only supports batch_size=1")

        all_logits = []

        for t in range(seq_len):
            token_id = input_ids[0, t].item()
            pos = positions[0, t].item() if positions is not None else self._position
            # The kernel writes KV at `position`, then attention sees [0, cache_len).
            # So cache_len must be position + 1 so the current token is visible.
            cache_len = pos + 1

            if t == seq_len - 1 and self.greedy_fast_path:
                # Fast path: kernel internal argmax only, no full logits writeback.
                # Return a compact [1, 1, 1] tensor holding the token id.
                # Sampler detects shape[-1] == 1 and treats it as pre-selected token.
                next_token = self._run_decode(token_id, pos, cache_len)
                compact = torch.tensor(
                    [[[float(next_token)]]],
                    dtype=torch.float32, device=self._hidden.device,
                )
                all_logits.append(compact)
            elif t == seq_len - 1:
                # Last token: capture logits
                _, logits = self._run_decode_with_logits(token_id, pos, cache_len)
                all_logits.append(logits)
            else:
                # Intermediate tokens: just update KV cache
                self._run_decode(token_id, pos, cache_len)

            self._position = pos + 1

        # Stack into [1, seq_len, vocab_size]
        if len(all_logits) == 1:
            logits = all_logits[0].unsqueeze(0).unsqueeze(0)
        else:
            logits = torch.stack(all_logits, dim=0).unsqueeze(0)

        return logits
