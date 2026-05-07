"""Extract and convert mini-vllm model weights for the CUDA megakernel backend.

The megakernel expects separate (non-fused) weights per layer, stacked into
contiguous tensors.  mini-vllm uses fused QKV and gate_up projections, so we
need to split them here.
"""

import struct
import torch


def extract_megakernel_weights(model):
    """Extract megakernel-compatible weights from a loaded mini-vllm model.

    Args:
        model: A loaded ``Qwen3ForCausalLM`` instance (weights already injected).

    Returns:
        dict with keys:
          - embed_weight, final_norm_weight, lm_head_weight
          - layer_weights_bytes (packed uint8 tensor of raw pointers)
          - Individual weight tensors kept alive (to prevent GC)
    """
    cfg = model.config
    layers = model.model.layers
    num_layers = cfg.num_hidden_layers

    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", hidden_size // num_q_heads)

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # Extract and split per-layer weights
    all_input_norm = []
    all_q = []
    all_k = []
    all_v = []
    all_q_norm = []
    all_k_norm = []
    all_o = []
    all_post_norm = []
    all_gate = []
    all_up = []
    all_down = []

    for layer in layers:
        # Split fused QKV: [q_size + kv_size + kv_size, hidden_size]
        qkv_w = layer.self_attn.qkv_proj.weight  # [out, in]
        q, k, v = qkv_w.split([q_size, kv_size, kv_size], dim=0)
        all_q.append(q)
        all_k.append(k)
        all_v.append(v)

        # Split fused gate_up: [2 * intermediate_size, hidden_size]
        gate_up_w = layer.mlp.gate_up_proj.weight  # [out, in]
        gate, up = gate_up_w.split([intermediate_size, intermediate_size], dim=0)
        all_gate.append(gate)
        all_up.append(up)

        all_input_norm.append(layer.input_layernorm.weight)
        all_o.append(layer.self_attn.o_proj.weight)
        all_post_norm.append(layer.post_attention_layernorm.weight)
        all_down.append(layer.mlp.down_proj.weight)

        # Q/K norm only exist when qkv_bias=False (Qwen3 default)
        if hasattr(layer.self_attn, "q_norm") and layer.self_attn.q_norm is not None:
            all_q_norm.append(layer.self_attn.q_norm.weight)
            all_k_norm.append(layer.self_attn.k_norm.weight)
        else:
            # Fallback: identity norm (weight = 1.0)
            all_q_norm.append(torch.ones(head_dim, device=qkv_w.device, dtype=qkv_w.dtype))
            all_k_norm.append(torch.ones(head_dim, device=qkv_w.device, dtype=qkv_w.dtype))

    # Stack per-layer weights into [num_layers, ...] tensors
    def stack(tensors, dtype=torch.bfloat16):
        return torch.stack([t.to(dtype).contiguous() for t in tensors])

    stacked = {
        "input_norm": stack(all_input_norm),
        "q": stack(all_q),
        "k": stack(all_k),
        "v": stack(all_v),
        "q_norm": stack(all_q_norm),
        "k_norm": stack(all_k_norm),
        "o": stack(all_o),
        "post_norm": stack(all_post_norm),
        "gate": stack(all_gate),
        "up": stack(all_up),
        "down": stack(all_down),
    }

    # Pack layer weight pointers into device-side struct array
    ptrs = []
    for i in range(num_layers):
        ptrs.extend([
            stacked["input_norm"][i].data_ptr(),
            stacked["q"][i].data_ptr(),
            stacked["k"][i].data_ptr(),
            stacked["v"][i].data_ptr(),
            stacked["q_norm"][i].data_ptr(),
            stacked["k_norm"][i].data_ptr(),
            stacked["o"][i].data_ptr(),
            stacked["post_norm"][i].data_ptr(),
            stacked["gate"][i].data_ptr(),
            stacked["up"][i].data_ptr(),
            stacked["down"][i].data_ptr(),
        ])

    raw = struct.pack(f"{len(ptrs)}Q", *ptrs)
    layer_weights_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(
        model.lm_head.weight.device
    )

    # Global weights
    embed_weight = model.model.embed_tokens.weight.to(torch.bfloat16).contiguous()
    final_norm_weight = model.model.norm.weight.to(torch.bfloat16).contiguous()

    # LM head (may be tied to embedding)
    lm_head_weight = model.lm_head.weight.to(torch.bfloat16).contiguous()

    return {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head_weight,
        "layer_weights_bytes": layer_weights_bytes,
        "stacked": stacked,
    }
