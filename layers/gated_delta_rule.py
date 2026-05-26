"""Triton-accelerated Gated Delta Rule kernels for Qwen3.5 decode path.

Fuses the small elementwise/reduction ops that dominate decode latency:
1. Causal conv1d update + SiLU  (replaces cat/conv1d/silu/copy_)
2. Recurrent gated delta rule   (replaces mul/sum/outer-product loop)
3. RMSNorm + SiLU gate          (replaces pow/mean/rsqrt/mul/silu)
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FLA (flash-linear-attention) backend for prefill
# ---------------------------------------------------------------------------

_FLA_AVAILABLE = False
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
    _FLA_AVAILABLE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# 1. Causal Conv1d Update (single-step decode)
# ---------------------------------------------------------------------------

@triton.jit
def _causal_conv1d_update_kernel(
    Hidden_ptr,
    State_ptr,
    Weight_ptr,
    Out_ptr,
    conv_dim,
    stride_hidden_c,
    stride_state_c,
    stride_weight_c,
    KERNEL_SIZE: tl.constexpr,
    STATE_LEN: tl.constexpr,
):
    """Single-step causal conv1d + SiLU, in-place state update.

    Input hidden shape:  (conv_dim,)  – one new value per channel
    State shape:         (conv_dim, STATE_LEN)
    Weight shape:        (conv_dim, KERNEL_SIZE)
    Output shape:        (conv_dim,)
    """
    channel = tl.program_id(0)

    # Compute conv: sum(state[i] * weight[i]) + hidden * weight[STATE_LEN]
    conv_val = tl.zeros((), dtype=tl.float32)
    for i in range(STATE_LEN):
        s = tl.load(State_ptr + channel * stride_state_c + i).to(tl.float32)
        w = tl.load(Weight_ptr + channel * stride_weight_c + i).to(tl.float32)
        conv_val += s * w

    hidden = tl.load(Hidden_ptr + channel * stride_hidden_c).to(tl.float32)
    w_last = tl.load(Weight_ptr + channel * stride_weight_c + STATE_LEN).to(tl.float32)
    conv_val += hidden * w_last

    # SiLU
    out_val = conv_val * tl.sigmoid(conv_val)
    tl.store(Out_ptr + channel, out_val)

    # Update state in-place: shift left, append hidden.
    # Read-ahead: for position i, read from i+1 and write to i.
    for i in range(STATE_LEN - 1):
        next_val = tl.load(State_ptr + channel * stride_state_c + i + 1)
        tl.store(State_ptr + channel * stride_state_c + i, next_val)
    tl.store(State_ptr + channel * stride_state_c + STATE_LEN - 1, hidden)


def triton_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Triton causal conv1d update for decode (batch=1, seq=1).

    Args:
        hidden_states: (batch=1, conv_dim, seq=1)
        conv_state:    (batch=1, conv_dim, state_len)
        weight:        (conv_dim, kernel_size)

    Returns:
        out: (batch=1, conv_dim, seq=1)
    """
    batch, conv_dim, seq_len = hidden_states.shape
    kernel_size = weight.shape[-1]
    assert batch == 1 and seq_len == 1
    assert conv_state.shape[-1] == kernel_size - 1

    hidden_flat = hidden_states.view(conv_dim)
    out = torch.empty_like(hidden_flat)

    grid = (conv_dim,)
    _causal_conv1d_update_kernel[grid](
        hidden_flat,
        conv_state.view(batch, conv_dim, kernel_size - 1),
        weight,
        out,
        conv_dim,
        hidden_flat.stride(0),
        conv_state.stride(1),
        weight.stride(0),
        KERNEL_SIZE=kernel_size,
        STATE_LEN=kernel_size - 1,
    )
    return out.view(batch, conv_dim, seq_len)


# ---------------------------------------------------------------------------
# 2. Recurrent Gated Delta Rule (single-step decode)
# ---------------------------------------------------------------------------

@triton.jit
def _recurrent_gated_delta_rule_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    G_ptr,
    Beta_ptr,
    State_ptr,
    Out_ptr,
    num_heads,
    head_k_dim,
    head_v_dim,
    scale,
    stride_state_h,
    stride_state_k,
    stride_state_v,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fused recurrent state update for gated delta rule (seq_len=1).

    Per head:
        state *= exp(g)
        kv_mem = sum_k(state[k,:] * k[k])          -- reduction over k
        delta  = (v - kv_mem) * beta
        state += k[:,None] * delta[None,:]          -- outer product
        output = sum_k(state[k,:] * q[k])          -- reduction over k
    """
    head_idx = tl.program_id(0)

    # Pointers for this head
    q_ptr = Q_ptr + head_idx * head_k_dim
    k_ptr = K_ptr + head_idx * head_k_dim
    v_ptr = V_ptr + head_idx * head_v_dim
    state_ptr = State_ptr + head_idx * stride_state_h
    out_ptr = Out_ptr + head_idx * head_v_dim

    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)

    # Load q, k, v vectors
    q = tl.load(q_ptr + k_offs, mask=k_offs < head_k_dim, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + k_offs, mask=k_offs < head_k_dim, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + v_offs, mask=v_offs < head_v_dim, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + head_idx).to(tl.float32)
    beta = tl.load(Beta_ptr + head_idx).to(tl.float32)

    # L2 normalise q and k (use_qk_l2norm_in_kernel=True)
    q_norm = tl.sqrt(tl.sum(q * q)) + 1e-6
    k_norm = tl.sqrt(tl.sum(k * k)) + 1e-6

    # exp(g)
    g = tl.exp(g)

    # Step 1: state *= g  (elementwise)
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            state_col = state_col * g
            tl.store(state_ptr + ki * stride_state_k + v_offs, state_col, mask=v_offs < head_v_dim)

    # Step 2: kv_mem[v] = sum_k state[k,v] * k[k]  (k is L2-normalised)
    kv_mem = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            k_val = tl.load(k_ptr + ki).to(tl.float32) / k_norm
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            kv_mem += state_col * k_val

    # Step 3: delta = (v - kv_mem) * beta
    delta = (v - kv_mem) * beta

    # Step 4: state += k[:,None] * delta[None,:]  (outer product, k normalised)
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            k_val = tl.load(k_ptr + ki).to(tl.float32) / k_norm
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            state_col = state_col + k_val * delta
            tl.store(state_ptr + ki * stride_state_k + v_offs, state_col, mask=v_offs < head_v_dim)

    # Step 5: output[v] = sum_k state[k,v] * q[k]  (q is L2-normalised + scaled)
    out = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            q_val = tl.load(q_ptr + ki).to(tl.float32) / q_norm * scale
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            out += state_col * q_val

    tl.store(out_ptr + v_offs, out, mask=v_offs < head_v_dim)


def triton_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Triton recurrent gated delta rule for decode (batch=1, seq=1).

    Args:
        query:  (batch=1, seq=1, num_heads, head_k_dim)
        key:    (batch=1, seq=1, num_heads, head_k_dim)
        value:  (batch=1, seq=1, num_heads, head_v_dim)
        g:      (batch=1, seq=1, num_heads)
        beta:   (batch=1, seq=1, num_heads)
        initial_state: (batch=1, num_heads, head_k_dim, head_v_dim) or None

    Returns:
        core_attn_out: (batch=1, seq=1, num_heads, head_v_dim)
        last_state:    (batch=1, num_heads, head_k_dim, head_v_dim) or None
    """
    batch, seq_len, num_heads, head_k_dim = query.shape
    head_v_dim = value.shape[-1]
    assert batch == 1 and seq_len == 1

    if scale is None:
        scale = 1.0 / (head_k_dim ** 0.5)

    # Flatten batch and seq for the kernel
    q = query.view(num_heads, head_k_dim)
    k = key.view(num_heads, head_k_dim)
    v = value.view(num_heads, head_v_dim)
    g_flat = g.view(num_heads)
    beta_flat = beta.view(num_heads)

    if initial_state is None:
        state = torch.zeros(
            num_heads, head_k_dim, head_v_dim,
            dtype=torch.float32, device=query.device,
        )
    else:
        state = initial_state.view(num_heads, head_k_dim, head_v_dim).to(torch.float32)

    out = torch.empty(num_heads, head_v_dim, dtype=torch.float32, device=query.device)

    grid = (num_heads,)
    BLOCK_K = triton.next_power_of_2(head_k_dim)
    BLOCK_V = triton.next_power_of_2(head_v_dim)

    _recurrent_gated_delta_rule_kernel[grid](
        q, k, v, g_flat, beta_flat,
        state, out,
        num_heads, head_k_dim, head_v_dim, scale,
        state.stride(0), state.stride(1), state.stride(2),
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    # Restore shapes
    out = out.to(query.dtype).view(batch, seq_len, num_heads, head_v_dim)
    state = state.view(batch, num_heads, head_k_dim, head_v_dim)

    if output_final_state:
        return out, state
    return out, None


# ---------------------------------------------------------------------------
# 2b. Recurrent Gated Delta Rule + RMSNorm Gated (fused decode)
# ---------------------------------------------------------------------------

@triton.jit
def _recurrent_norm_gated_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    G_ptr,
    Beta_ptr,
    State_ptr,
    Z_ptr,
    NormWeight_ptr,
    Out_ptr,
    num_heads,
    head_k_dim,
    head_v_dim,
    scale,
    norm_eps,
    stride_state_h,
    stride_state_k,
    stride_state_v,
    stride_z_h,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fused recurrent delta rule + RMSNorm gated (seq_len=1).

    Same as _recurrent_gated_delta_rule_kernel but appends RMSNorm+SiLU
    before writing the final output, avoiding a round-trip through global
    memory between recurrent and norm.
    """
    head_idx = tl.program_id(0)

    q_ptr = Q_ptr + head_idx * head_k_dim
    k_ptr = K_ptr + head_idx * head_k_dim
    v_ptr = V_ptr + head_idx * head_v_dim
    state_ptr = State_ptr + head_idx * stride_state_h
    z_ptr = Z_ptr + head_idx * stride_z_h
    out_ptr = Out_ptr + head_idx * head_v_dim

    k_offs = tl.arange(0, BLOCK_K)
    v_offs = tl.arange(0, BLOCK_V)

    q = tl.load(q_ptr + k_offs, mask=k_offs < head_k_dim, other=0.0).to(tl.float32)
    k = tl.load(k_ptr + k_offs, mask=k_offs < head_k_dim, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + v_offs, mask=v_offs < head_v_dim, other=0.0).to(tl.float32)
    g = tl.load(G_ptr + head_idx).to(tl.float32)
    beta = tl.load(Beta_ptr + head_idx).to(tl.float32)

    q_norm = tl.sqrt(tl.sum(q * q)) + 1e-6
    k_norm = tl.sqrt(tl.sum(k * k)) + 1e-6

    g = tl.exp(g)

    # Step 1: state *= g
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            state_col = state_col * g
            tl.store(state_ptr + ki * stride_state_k + v_offs, state_col, mask=v_offs < head_v_dim)

    # Step 2: kv_mem = sum_k state[k,:] * k[k]
    kv_mem = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            k_val = tl.load(k_ptr + ki).to(tl.float32) / k_norm
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            kv_mem += state_col * k_val

    # Step 3: delta = (v - kv_mem) * beta
    delta = (v - kv_mem) * beta

    # Step 4: state += k[:,None] * delta[None,:]
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            k_val = tl.load(k_ptr + ki).to(tl.float32) / k_norm
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            state_col = state_col + k_val * delta
            tl.store(state_ptr + ki * stride_state_k + v_offs, state_col, mask=v_offs < head_v_dim)

    # Step 5: output = sum_k state[k,:] * q[k]
    out = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for ki in range(BLOCK_K):
        if ki < head_k_dim:
            q_val = tl.load(q_ptr + ki).to(tl.float32) / q_norm * scale
            state_col = tl.load(state_ptr + ki * stride_state_k + v_offs, mask=v_offs < head_v_dim, other=0.0)
            out += state_col * q_val

    # Step 6: RMSNorm + SiLU gate (in-place on `out`)
    z = tl.load(z_ptr + v_offs, mask=v_offs < head_v_dim, other=0.0).to(tl.float32)
    norm_w = tl.load(NormWeight_ptr + v_offs, mask=v_offs < head_v_dim, other=0.0).to(tl.float32)

    var = tl.sum(out * out) / head_v_dim
    rrms = 1.0 / tl.sqrt(var + norm_eps)

    silu_z = z * tl.sigmoid(z)
    final_out = out * rrms * norm_w * silu_z

    tl.store(out_ptr + v_offs, final_out.to(tl.bfloat16), mask=v_offs < head_v_dim)


def triton_recurrent_norm_gated(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: float | None = None,
    z: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Triton fused recurrent gated delta rule + RMSNorm gated for decode.

    Args:
        query:  (batch=1, seq=1, num_heads, head_k_dim)
        key:    (batch=1, seq=1, num_heads, head_k_dim)
        value:  (batch=1, seq=1, num_heads, head_v_dim)
        g:      (batch=1, seq=1, num_heads)
        beta:   (batch=1, seq=1, num_heads)
        initial_state: (batch=1, num_heads, head_k_dim, head_v_dim) or None
        z:      (batch=1, num_heads, head_v_dim)
        norm_weight: (head_v_dim,)
        norm_eps: float

    Returns:
        core_attn_out: (batch=1, seq=1, num_heads, head_v_dim)
        last_state:    (batch=1, num_heads, head_k_dim, head_v_dim) or None
    """
    batch, seq_len, num_heads, head_k_dim = query.shape
    head_v_dim = value.shape[-1]
    assert batch == 1 and seq_len == 1
    assert z is not None and norm_weight is not None

    if scale is None:
        scale = 1.0 / (head_k_dim ** 0.5)

    q = query.view(num_heads, head_k_dim)
    k = key.view(num_heads, head_k_dim)
    v = value.view(num_heads, head_v_dim)
    g_flat = g.view(num_heads)
    beta_flat = beta.view(num_heads)
    z_flat = z.view(num_heads, head_v_dim)

    if initial_state is None:
        state = torch.zeros(
            num_heads, head_k_dim, head_v_dim,
            dtype=torch.float32, device=query.device,
        )
    else:
        state = initial_state.view(num_heads, head_k_dim, head_v_dim).to(torch.float32)

    out = torch.empty(num_heads, head_v_dim, dtype=torch.bfloat16, device=query.device)

    grid = (num_heads,)
    BLOCK_K = triton.next_power_of_2(head_k_dim)
    BLOCK_V = triton.next_power_of_2(head_v_dim)

    _recurrent_norm_gated_kernel[grid](
        q, k, v, g_flat, beta_flat,
        state, z_flat, norm_weight, out,
        num_heads, head_k_dim, head_v_dim, scale, norm_eps,
        state.stride(0), state.stride(1), state.stride(2),
        z_flat.stride(0),
        BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
    )

    out = out.view(batch, seq_len, num_heads, head_v_dim)
    state = state.view(batch, num_heads, head_k_dim, head_v_dim)

    if output_final_state:
        return out, state
    return out, None


# ---------------------------------------------------------------------------
# 3. RMSNorm + SiLU Gate
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_gated_kernel(
    X_ptr,
    Gate_ptr,
    W_ptr,
    Out_ptr,
    M,
    N,
    eps,
    stride_x_m,
    stride_x_n,
    stride_gate_m,
    stride_gate_n,
    stride_out_m,
    stride_out_n,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm + SiLU(gate).

    out = (x / sqrt(mean(x^2) + eps)) * w * silu(gate)
    """
    row = tl.program_id(0)

    x_ptr = X_ptr + row * stride_x_m
    gate_ptr = Gate_ptr + row * stride_gate_m
    out_ptr = Out_ptr + row * stride_out_m

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + cols * stride_gate_n, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)

    silu_gate = gate * tl.sigmoid(gate)
    out = x * rrms * w * silu_gate

    tl.store(out_ptr + cols * stride_out_n, out, mask=mask)


def triton_rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Triton RMSNorm + SiLU gate.

    Args:
        x:      (M, N)
        gate:   (M, N)
        weight: (N,)

    Returns:
        out: (M, N)
    """
    M, N = x.shape
    out = torch.empty_like(x)

    grid = (M,)
    BLOCK_SIZE = triton.next_power_of_2(N)

    _rms_norm_gated_kernel[grid](
        x, gate, weight, out,
        M, N, eps,
        x.stride(0), x.stride(1),
        gate.stride(0), gate.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class TritonRMSNormGated(nn.Module):
    """RMSNorm with SiLU gate – Triton fused decode path."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x_flat = hidden_states.reshape(-1, orig_shape[-1])
        gate_flat = gate.reshape(-1, orig_shape[-1])
        out_flat = triton_rms_norm_gated(x_flat, gate_flat, self.weight, self.variance_epsilon)
        return out_flat.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Fused decode kernel: conv1d + recurrent + norm in one launch
# ---------------------------------------------------------------------------

@triton.jit
def _fused_linear_attn_decode_kernel(
    MixedQKV_ptr,
    ConvState_ptr,
    ConvWeight_ptr,
    B_ptr,
    A_ptr,
    A_log_ptr,
    dt_bias_ptr,
    RecurrentState_ptr,
    Z_ptr,
    NormWeight_ptr,
    Out_ptr,
    num_heads,
    head_k_dim,
    head_v_dim,
    scale,
    norm_eps,
    stride_state_c,
    stride_weight_c,
    stride_recurrent_h,
    stride_recurrent_k,
    stride_recurrent_v,
    stride_z_h,
    stride_out_h,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    """Single-step decode for one head: causal conv1d + recurrent delta rule + RMSNorm gated.

    Grid: (num_v_heads,).  Each program processes one head.
    Block size = BLOCK_C (= head_k_dim, typically 128).
    """
    head_idx = tl.program_id(0)
    tid = tl.arange(0, BLOCK_C)

    # Offsets for this head's q/k/v channels in the conv buffer
    q_base = head_idx * BLOCK_C
    k_base = num_heads * BLOCK_C + head_idx * BLOCK_C
    v_base = 2 * num_heads * BLOCK_C + head_idx * BLOCK_C

    # ------------------------------------------------------------------
    # Phase 1: Causal conv1d (3 rounds: q, k, v – 128 channels each)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Phase 1: Causal conv1d (3 rounds: q, k, v – 128 channels each)
    # We overwrite MixedQKV_ptr in-place with the SiLU(conv) output so that
    # the recurrent phase can load individual elements via tl.load.
    # ------------------------------------------------------------------
    # Round 0 – q channels
    channels = q_base + tid
    s0 = tl.load(ConvState_ptr + channels * stride_state_c + 0)
    s1 = tl.load(ConvState_ptr + channels * stride_state_c + 1)
    s2 = tl.load(ConvState_ptr + channels * stride_state_c + 2)
    w0 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 0)
    w1 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 1)
    w2 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 2)
    w3 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 3)
    h = tl.load(MixedQKV_ptr + channels)
    conv_val = (
        s0.to(tl.float32) * w0.to(tl.float32)
        + s1.to(tl.float32) * w1.to(tl.float32)
        + s2.to(tl.float32) * w2.to(tl.float32)
        + h.to(tl.float32) * w3.to(tl.float32)
    )
    out_val = conv_val * tl.sigmoid(conv_val)
    tl.store(MixedQKV_ptr + channels, out_val.to(tl.bfloat16))
    tl.store(ConvState_ptr + channels * stride_state_c + 0, s1)
    tl.store(ConvState_ptr + channels * stride_state_c + 1, s2)
    tl.store(ConvState_ptr + channels * stride_state_c + 2, h)

    # Round 1 – k channels
    channels = k_base + tid
    s0 = tl.load(ConvState_ptr + channels * stride_state_c + 0)
    s1 = tl.load(ConvState_ptr + channels * stride_state_c + 1)
    s2 = tl.load(ConvState_ptr + channels * stride_state_c + 2)
    w0 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 0)
    w1 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 1)
    w2 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 2)
    w3 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 3)
    h = tl.load(MixedQKV_ptr + channels)
    conv_val = (
        s0.to(tl.float32) * w0.to(tl.float32)
        + s1.to(tl.float32) * w1.to(tl.float32)
        + s2.to(tl.float32) * w2.to(tl.float32)
        + h.to(tl.float32) * w3.to(tl.float32)
    )
    out_val = conv_val * tl.sigmoid(conv_val)
    tl.store(MixedQKV_ptr + channels, out_val.to(tl.bfloat16))
    tl.store(ConvState_ptr + channels * stride_state_c + 0, s1)
    tl.store(ConvState_ptr + channels * stride_state_c + 1, s2)
    tl.store(ConvState_ptr + channels * stride_state_c + 2, h)

    # Round 2 – v channels
    channels = v_base + tid
    s0 = tl.load(ConvState_ptr + channels * stride_state_c + 0)
    s1 = tl.load(ConvState_ptr + channels * stride_state_c + 1)
    s2 = tl.load(ConvState_ptr + channels * stride_state_c + 2)
    w0 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 0)
    w1 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 1)
    w2 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 2)
    w3 = tl.load(ConvWeight_ptr + channels * stride_weight_c + 3)
    h = tl.load(MixedQKV_ptr + channels)
    conv_val = (
        s0.to(tl.float32) * w0.to(tl.float32)
        + s1.to(tl.float32) * w1.to(tl.float32)
        + s2.to(tl.float32) * w2.to(tl.float32)
        + h.to(tl.float32) * w3.to(tl.float32)
    )
    out_val = conv_val * tl.sigmoid(conv_val)
    tl.store(MixedQKV_ptr + channels, out_val.to(tl.bfloat16))
    tl.store(ConvState_ptr + channels * stride_state_c + 0, s1)
    tl.store(ConvState_ptr + channels * stride_state_c + 1, s2)
    tl.store(ConvState_ptr + channels * stride_state_c + 2, h)

    # ------------------------------------------------------------------
    # Phase 2: Recurrent gated delta rule
    # ------------------------------------------------------------------
    b = tl.load(B_ptr + head_idx).to(tl.float32)
    a = tl.load(A_ptr + head_idx).to(tl.float32)
    a_log = tl.load(A_log_ptr + head_idx).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + head_idx).to(tl.float32)

    beta = tl.sigmoid(b)
    # stable softplus: log(1+exp(x))
    softplus_a_dt = tl.where(a + dt_bias > 20, a + dt_bias, tl.log(1.0 + tl.exp(a + dt_bias)))
    g = -tl.exp(a_log) * softplus_a_dt
    gate = tl.exp(g)

    q_vec = tl.load(MixedQKV_ptr + q_base + tid).to(tl.float32)
    k_vec = tl.load(MixedQKV_ptr + k_base + tid).to(tl.float32)
    q_norm = tl.sqrt(tl.sum(q_vec * q_vec)) + 1e-6
    k_norm = tl.sqrt(tl.sum(k_vec * k_vec)) + 1e-6

    state_ptr = RecurrentState_ptr + head_idx * stride_recurrent_h
    v_offs = tl.arange(0, BLOCK_V)

    # Step 1: state *= gate
    for ki in range(BLOCK_K):
        state_col = tl.load(state_ptr + ki * stride_recurrent_k + v_offs)
        state_col = state_col * gate
        tl.store(state_ptr + ki * stride_recurrent_k + v_offs, state_col)

    # Step 2: kv_mem[v] = sum_k state[k,v] * k[k] / k_norm
    kv_mem = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for ki in range(BLOCK_K):
        k_val = tl.load(MixedQKV_ptr + k_base + ki).to(tl.float32) / k_norm
        state_col = tl.load(state_ptr + ki * stride_recurrent_k + v_offs)
        kv_mem += state_col * k_val

    # Step 3: delta = (v - kv_mem) * beta
    v_vec = tl.load(MixedQKV_ptr + v_base + tid).to(tl.float32)
    delta = (v_vec - kv_mem) * beta

    # Step 4: state += k[:,None] * delta[None,:] / k_norm
    for ki in range(BLOCK_K):
        k_val = tl.load(MixedQKV_ptr + k_base + ki).to(tl.float32) / k_norm
        state_col = tl.load(state_ptr + ki * stride_recurrent_k + v_offs)
        state_col = state_col + k_val * delta
        tl.store(state_ptr + ki * stride_recurrent_k + v_offs, state_col)

    # Step 5: output[v] = sum_k state[k,v] * q[k] / q_norm * scale
    out = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for ki in range(BLOCK_K):
        q_val = tl.load(MixedQKV_ptr + q_base + ki).to(tl.float32) / q_norm * scale
        state_col = tl.load(state_ptr + ki * stride_recurrent_k + v_offs)
        out += state_col * q_val

    # ------------------------------------------------------------------
    # Phase 3: RMSNorm + SiLU gate
    # ------------------------------------------------------------------
    z = tl.load(Z_ptr + head_idx * stride_z_h + v_offs).to(tl.float32)
    norm_w = tl.load(NormWeight_ptr + v_offs).to(tl.float32)

    var = tl.sum(out * out) / BLOCK_V
    rrms = 1.0 / tl.sqrt(var + norm_eps)

    silu_z = z * tl.sigmoid(z)
    final_out = out * rrms * norm_w * silu_z

    tl.store(Out_ptr + head_idx * stride_out_h + v_offs, final_out.to(tl.bfloat16))


def triton_fused_linear_attn_decode(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused single-step decode for linear attention (conv1d + recurrent + norm).

    Args:
        mixed_qkv:  (batch=1, conv_dim, seq=1) – already transposed for conv1d
        conv_state: (batch=1, conv_dim, state_len)
        conv_weight:(conv_dim, kernel_size)
        b:          (batch=1, num_v_heads)
        a:          (batch=1, num_v_heads)
        A_log:      (num_v_heads,)
        dt_bias:    (num_v_heads,)
        initial_state: (batch=1, num_v_heads, head_k_dim, head_v_dim) or None
        z:          (batch=1, num_v_heads, head_v_dim)
        norm_weight:(head_v_dim,)
        norm_eps:   float

    Returns:
        core_attn_out: (batch=1, 1, num_v_heads, head_v_dim)
        last_state:    (batch=1, num_v_heads, head_k_dim, head_v_dim)
    """
    batch, conv_dim, seq_len = mixed_qkv.shape
    num_v_heads = z.shape[1]
    head_k_dim = num_v_heads  # Actually we need to infer from state; but state may be None
    # Wait, head_k_dim and head_v_dim should come from initial_state or config
    # Let's use the conv_weight shape and mixed_qkv shape to infer
    # conv_dim = key_dim * 2 + value_dim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    # For Qwen3.5, num_k_heads = num_v_heads, head_k_dim = head_v_dim
    # So conv_dim = num_v_heads * head_k_dim * 3
    # Therefore head_k_dim = conv_dim // (3 * num_v_heads)
    head_k_dim = conv_dim // (3 * num_v_heads)
    head_v_dim = head_k_dim

    assert batch == 1 and seq_len == 1
    assert conv_dim == num_v_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    if scale is None:
        scale = 1.0 / (head_k_dim ** 0.5)

    mixed_qkv_flat = mixed_qkv.view(conv_dim)
    conv_state_flat = conv_state.view(batch, conv_dim, conv_weight.shape[-1] - 1)
    b_flat = b.view(num_v_heads)
    a_flat = a.view(num_v_heads)
    z_flat = z.view(num_v_heads, head_v_dim)

    if initial_state is None:
        state = torch.zeros(
            num_v_heads, head_k_dim, head_v_dim,
            dtype=torch.float32, device=mixed_qkv.device,
        )
    else:
        state = initial_state.view(num_v_heads, head_k_dim, head_v_dim).to(torch.float32)

    out = torch.empty(num_v_heads, head_v_dim, dtype=torch.bfloat16, device=mixed_qkv.device)

    grid = (num_v_heads,)
    BLOCK_C = triton.next_power_of_2(head_k_dim)
    BLOCK_K = triton.next_power_of_2(head_k_dim)
    BLOCK_V = triton.next_power_of_2(head_v_dim)

    _fused_linear_attn_decode_kernel[grid](
        mixed_qkv_flat,
        conv_state_flat,
        conv_weight,
        b_flat,
        a_flat,
        A_log,
        dt_bias,
        state,
        z_flat,
        norm_weight,
        out,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        scale,
        norm_eps,
        conv_state_flat.stride(1),
        conv_weight.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        z_flat.stride(0),
        out.stride(0),
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        STATE_LEN=conv_weight.shape[-1] - 1,
        KERNEL_SIZE=conv_weight.shape[-1],
    )

    out = out.view(batch, 1, num_v_heads, head_v_dim)
    state = state.view(batch, num_v_heads, head_k_dim, head_v_dim)
    return out, state


# ---------------------------------------------------------------------------
# Torch fallbacks (no Triton / no causal_conv1d_cuda)
# ---------------------------------------------------------------------------


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
):
    """Single-step causal conv1d update (torch fallback)."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size
    )
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Recurrent gated delta rule for single-token decode (torch fallback)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size,
        num_heads,
        sequence_length,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = (
            last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        )
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Chunked gated delta rule for prefill (torch fallback)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(
        chunk_size, dtype=attn.dtype, device=attn.device
    )
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_local = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_local @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (
                k_i
                * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _prefill_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    backend: str = "torch",
):
    """Prefill path for gated delta rule with pluggable backend.

    Args:
        backend: "fla" (flash-linear-attention Triton kernel) or "torch".
    """
    if backend == "fla" and _FLA_AVAILABLE:
        try:
            return _fla_chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        except Exception as e:
            logger.warning(
                "FLA chunk_gated_delta_rule failed (%s), falling back to torch", e
            )

    # torch path: prefer chunked, fall back to recurrent
    try:
        return _torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    except Exception as e:
        logger.warning(
            "Torch chunked gated delta rule failed (%s), falling back to recurrent loop", e
        )
        return _torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
