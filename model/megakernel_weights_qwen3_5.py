"""Extract and convert Qwen3.5 model weights for the megakernel backend.

Qwen3.5 has a hybrid architecture (linear + full attention), so weight
extraction must handle both layer types.
"""

import struct
import torch


def extract_megakernel_weights_qwen3_5(model):
    """Extract megakernel-compatible weights from a loaded Qwen3.5 model.

    Args:
        model: A loaded ``Qwen3_5ForCausalLM`` instance (weights already injected).

    Returns:
        dict with keys:
          - embed_weight, final_norm_weight, lm_head_weight
          - layer_weights_bytes (packed uint8 tensor of raw pointers)
          - Per-layer weight tensors kept alive (to prevent GC)
    """
    cfg = model.config
    layers = model.model.language_model.layers
    num_layers = cfg.num_hidden_layers

    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size

    # Full attention dims
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", hidden_size // num_q_heads)
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # Linear attention dims
    num_k_heads = cfg.linear_num_key_heads
    num_v_heads = cfg.linear_num_value_heads
    head_k_dim = cfg.linear_key_head_dim
    head_v_dim = cfg.linear_value_head_dim
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    conv_kernel_size = cfg.linear_conv_kernel_dim

    # Extract per-layer weights
    all_input_norm = []
    all_post_norm = []
    all_gate = []
    all_up = []
    all_down = []

    # Full attention specific
    all_full_q = []
    all_full_k = []
    all_full_v = []
    all_full_q_norm = []
    all_full_k_norm = []
    all_full_o = []

    # Linear attention specific
    all_linear_in_proj_qkv = []
    all_linear_conv1d_weight = []
    all_linear_in_proj_z = []
    all_linear_in_proj_b = []
    all_linear_in_proj_a = []
    all_linear_dt_bias = []
    all_linear_a_log = []
    all_linear_norm_weight = []
    all_linear_out_proj = []

    layer_types = []

    for layer in layers:
        layer_type = layer.layer_type
        layer_types.append(layer_type)

        # Common weights (both layer types have these)
        all_input_norm.append(layer.input_layernorm.weight)
        all_post_norm.append(layer.post_attention_layernorm.weight)
        all_down.append(layer.mlp.down_proj.weight)

        # MLP: gate_up_proj merged -> split for megakernel
        gate_up_w = layer.mlp.gate_up_proj.weight
        all_gate.append(gate_up_w[:intermediate_size, :])
        all_up.append(gate_up_w[intermediate_size:, :])

        if layer_type == "full_attention":
            attn = layer.self_attn
            # qkv_gate_proj merged -> split for megakernel
            # q_proj in megakernel expects [q+gate, hidden] (Q_SIZE*2)
            qkv_gate_w = attn.qkv_gate_proj.weight
            all_full_q.append(qkv_gate_w[: q_size * 2, :])  # q+gate
            all_full_k.append(qkv_gate_w[q_size * 2 : q_size * 2 + kv_size, :])
            all_full_v.append(qkv_gate_w[q_size * 2 + kv_size :, :])
            all_full_o.append(attn.o_proj.weight)

            if hasattr(attn, "q_norm") and attn.q_norm is not None:
                all_full_q_norm.append(attn.q_norm.weight)
                all_full_k_norm.append(attn.k_norm.weight)
            else:
                dev = attn.qkv_gate_proj.weight.device
                dt = attn.qkv_gate_proj.weight.dtype
                all_full_q_norm.append(torch.ones(head_dim, device=dev, dtype=dt))
                all_full_k_norm.append(torch.ones(head_dim, device=dev, dtype=dt))

            # Placeholders for linear weights (to keep indexing consistent)
            dev = attn.qkv_gate_proj.weight.device
            dt = attn.qkv_gate_proj.weight.dtype
            all_linear_in_proj_qkv.append(torch.zeros(conv_dim, hidden_size, device=dev, dtype=dt))
            all_linear_conv1d_weight.append(torch.zeros(conv_dim, conv_kernel_size, device=dev, dtype=dt))
            all_linear_in_proj_z.append(torch.zeros(value_dim, hidden_size, device=dev, dtype=dt))
            all_linear_in_proj_b.append(torch.zeros(num_v_heads, hidden_size, device=dev, dtype=dt))
            all_linear_in_proj_a.append(torch.zeros(num_v_heads, hidden_size, device=dev, dtype=dt))
            all_linear_dt_bias.append(torch.zeros(num_v_heads, device=dev, dtype=torch.float32))
            all_linear_a_log.append(torch.zeros(num_v_heads, device=dev, dtype=torch.float32))
            all_linear_norm_weight.append(torch.ones(head_v_dim, device=dev, dtype=dt))
            all_linear_out_proj.append(torch.zeros(hidden_size, value_dim, device=dev, dtype=dt))

        else:  # linear_attention
            attn = layer.linear_attn
            # in_proj merged -> split for megakernel
            in_proj_w = attn.in_proj.weight
            all_linear_in_proj_qkv.append(in_proj_w[:conv_dim, :])
            all_linear_in_proj_z.append(in_proj_w[conv_dim : conv_dim + value_dim, :])
            all_linear_in_proj_b.append(in_proj_w[conv_dim + value_dim : conv_dim + value_dim + num_v_heads, :])
            all_linear_in_proj_a.append(in_proj_w[conv_dim + value_dim + num_v_heads :, :])
            # conv1d weight: [conv_dim, 1, kernel_size] -> [conv_dim, kernel_size]
            all_linear_conv1d_weight.append(attn.conv1d.weight.squeeze(1))
            all_linear_dt_bias.append(attn.dt_bias)
            all_linear_a_log.append(attn.A_log)
            all_linear_norm_weight.append(attn.norm.weight)
            all_linear_out_proj.append(attn.out_proj.weight)

            # Placeholders for full attention weights
            dev = attn.in_proj.weight.device
            dt = attn.in_proj.weight.dtype
            all_full_q.append(torch.zeros(q_size * 2, hidden_size, device=dev, dtype=dt))
            all_full_k.append(torch.zeros(kv_size, hidden_size, device=dev, dtype=dt))
            all_full_v.append(torch.zeros(kv_size, hidden_size, device=dev, dtype=dt))
            all_full_q_norm.append(torch.ones(head_dim, device=dev, dtype=dt))
            all_full_k_norm.append(torch.ones(head_dim, device=dev, dtype=dt))
            all_full_o.append(torch.zeros(hidden_size, q_size, device=dev, dtype=dt))

    # Stack per-layer weights into [num_layers, ...] tensors
    def stack(tensors, dtype=torch.bfloat16):
        # Handle different dtypes (e.g. dt_bias and A_log are float32)
        if tensors and tensors[0].dtype == torch.float32:
            return torch.stack([t.to(torch.float32).contiguous() for t in tensors])
        return torch.stack([t.to(dtype).contiguous() for t in tensors])

    stacked = {
        "input_norm": stack(all_input_norm),
        "post_norm": stack(all_post_norm),
        "gate": stack(all_gate),
        "up": stack(all_up),
        "down": stack(all_down),
        "full_q": stack(all_full_q),
        "full_k": stack(all_full_k),
        "full_v": stack(all_full_v),
        "full_q_norm": stack(all_full_q_norm),
        "full_k_norm": stack(all_full_k_norm),
        "full_o": stack(all_full_o),
        "linear_in_proj_qkv": stack(all_linear_in_proj_qkv),
        "linear_conv1d_weight": stack(all_linear_conv1d_weight),
        "linear_in_proj_z": stack(all_linear_in_proj_z),
        "linear_in_proj_b": stack(all_linear_in_proj_b),
        "linear_in_proj_a": stack(all_linear_in_proj_a),
        "linear_dt_bias": stack(all_linear_dt_bias),
        "linear_a_log": stack(all_linear_a_log),
        "linear_norm_weight": stack(all_linear_norm_weight),
        "linear_out_proj": stack(all_linear_out_proj),
    }

    # Pack per-layer weight pointers into a unified struct array.
    # LayerWeights struct (C++ side):
    #   int64_t layer_type;   // 0 = full, 1 = linear
    #   const __nv_bfloat16 *input_norm;
    #   const __nv_bfloat16 *post_norm;
    #   const __nv_bfloat16 *gate_proj;
    #   const __nv_bfloat16 *up_proj;
    #   const __nv_bfloat16 *down_proj;
    #   // Full attention (unused -> nullptr for linear layers)
    #   const __nv_bfloat16 *full_q_proj;
    #   const __nv_bfloat16 *full_k_proj;
    #   const __nv_bfloat16 *full_v_proj;
    #   const __nv_bfloat16 *full_q_norm;
    #   const __nv_bfloat16 *full_k_norm;
    #   const __nv_bfloat16 *full_o_proj;
    #   // Linear attention (unused -> nullptr for full layers)
    #   const __nv_bfloat16 *linear_in_proj_qkv;
    #   const __nv_bfloat16 *linear_conv1d_weight;
    #   const __nv_bfloat16 *linear_in_proj_z;
    #   const __nv_bfloat16 *linear_in_proj_b;
    #   const __nv_bfloat16 *linear_in_proj_a;
    #   const float *linear_dt_bias;
    #   const float *linear_a_log;
    #   const __nv_bfloat16 *linear_norm_weight;
    #   const __nv_bfloat16 *linear_out_proj;
    # Total: 21 x 8 = 168 bytes per layer
    device = model.lm_head.weight.device
    all_layer_ptrs = []

    for i in range(num_layers):
        is_linear = 1 if layer_types[i] == "linear_attention" else 0
        ptrs = [
            is_linear,
            stacked["input_norm"][i].data_ptr(),
            stacked["post_norm"][i].data_ptr(),
            stacked["gate"][i].data_ptr(),
            stacked["up"][i].data_ptr(),
            stacked["down"][i].data_ptr(),
            # Full attention weights (0 if linear)
            stacked["full_q"][i].data_ptr() if not is_linear else 0,
            stacked["full_k"][i].data_ptr() if not is_linear else 0,
            stacked["full_v"][i].data_ptr() if not is_linear else 0,
            stacked["full_q_norm"][i].data_ptr() if not is_linear else 0,
            stacked["full_k_norm"][i].data_ptr() if not is_linear else 0,
            stacked["full_o"][i].data_ptr() if not is_linear else 0,
            # Linear attention weights (0 if full)
            stacked["linear_in_proj_qkv"][i].data_ptr() if is_linear else 0,
            stacked["linear_conv1d_weight"][i].data_ptr() if is_linear else 0,
            stacked["linear_in_proj_z"][i].data_ptr() if is_linear else 0,
            stacked["linear_in_proj_b"][i].data_ptr() if is_linear else 0,
            stacked["linear_in_proj_a"][i].data_ptr() if is_linear else 0,
            stacked["linear_dt_bias"][i].data_ptr() if is_linear else 0,
            stacked["linear_a_log"][i].data_ptr() if is_linear else 0,
            stacked["linear_norm_weight"][i].data_ptr() if is_linear else 0,
            stacked["linear_out_proj"][i].data_ptr() if is_linear else 0,
        ]
        all_layer_ptrs.extend(ptrs)

    raw = struct.pack(f"q{len(all_layer_ptrs) - 1}Q", *all_layer_ptrs)
    layer_weights_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)

    # Global weights
    embed_weight = model.model.language_model.embed_tokens.weight.to(torch.bfloat16).contiguous()
    final_norm_weight = model.model.language_model.norm.weight.to(torch.bfloat16).contiguous()
    lm_head_weight = model.lm_head.weight.to(torch.bfloat16).contiguous()

    return {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head_weight,
        "layer_weights_bytes": layer_weights_bytes,
        "stacked": stacked,
        "layer_types": layer_types,
    }
