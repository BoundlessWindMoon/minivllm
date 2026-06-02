"""Fused KIVI decode attention kernel (Triton).

Fuses the entire decode attention into two kernel launches:
  1. Per-tile partial attention with online softmax
  2. Cross-tile reduction to final output

This eliminates the 7 separate kernel launches of the old
`_compute_kivi_decode` path (Q×K_quant, Q×K_full, cat, softmax,
attn×V_quant, attn×V_full, add).

Supported:
  - 2-bit and 4-bit asymmetric KIVI quantization
  - GQA (grouped-query attention) via head replication inside the kernel
  - bf16 / fp16 input, fp32 accumulation
"""

import math

import torch
import triton
import triton.language as tl

from kernels.kivi.matmul import kivi_bmm_fA_qB_outer


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def kivi_decode_partial_kernel(
    q_ptr,
    k_code_ptr, k_scale_ptr, k_mn_ptr, k_full_ptr,
    v_code_ptr, v_scale_ptr, v_mn_ptr, v_full_ptr,
    partial_o_ptr, partial_m_ptr, partial_l_ptr,
    scale_factor,
    B, nh, nh_kv, D, T_k_quant, T_v_quant, total_len,
    stride_qb, stride_qh, stride_qd,
    stride_kcb, stride_kch, stride_kcd, stride_kct,
    stride_ksb, stride_ksh, stride_ksd, stride_ksg,
    stride_kmb, stride_kmh, stride_kmd, stride_kmg,
    stride_kfb, stride_kfh, stride_kft, stride_kfd,
    stride_vcb, stride_vch, stride_vct, stride_vcp,
    stride_vsb, stride_vsh, stride_vst, stride_vsg,
    stride_vmb, stride_vmh, stride_vmt, stride_vmg,
    stride_vfb, stride_vfh, stride_vft, stride_vfd,
    stride_pob, stride_poh, stride_pot, stride_pod,
    stride_pmb, stride_pmh, stride_pmt,
    bits: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute partial attention for one T tile.

    Each program handles one (batch, query_head, tile).
    Online softmax is used so that partial results can be reduced later.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_tile = tl.program_id(2)

    n_rep = nh // nh_kv
    pid_h_kv = pid_h // n_rep

    tile_start = pid_tile * BLOCK_T
    tile_end = tl.minimum(tile_start + BLOCK_T, total_len)

    # ---- Load Q ----
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_d < D
    q = tl.load(
        q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd,
        mask=q_mask, other=0.0,
    ).to(tl.float32)

    # ---- Online softmax state ----
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    feat_per_int = 32 // bits
    num = (0xFF >> (8 - bits))

    # ---- Iterate over positions in this tile ----
    for t in range(tile_start, tile_end):
        # ====== Load K_t ======
        if t < T_k_quant:
            t_group = t // group_size
            # scale / mn: (B, nh_kv, D, T_k_quant//group_size)
            sm_base = (
                pid_b * stride_ksb
                + pid_h_kv * stride_ksh
                + t_group * stride_ksg
            )
            scale = tl.load(
                k_scale_ptr + sm_base + offs_d * stride_ksd,
                mask=q_mask, other=0.0,
            )
            mn = tl.load(
                k_mn_ptr + sm_base + offs_d * stride_kmd,
                mask=q_mask, other=0.0,
            )

            t_packed = t // feat_per_int
            t_bit = (t % feat_per_int) * bits
            k_base = (
                pid_b * stride_kcb
                + pid_h_kv * stride_kch
                + t_packed * stride_kct
            )
            k_code = tl.load(
                k_code_ptr + k_base + offs_d * stride_kcd,
                mask=q_mask, other=0,
            )
            k_val = (
                ((k_code >> t_bit) & num).to(tl.float32) * scale + mn
            )
        else:
            t_k_full = t - T_k_quant
            k_base = (
                pid_b * stride_kfb
                + pid_h_kv * stride_kfh
                + t_k_full * stride_kft
            )
            k_val = tl.load(
                k_full_ptr + k_base + offs_d * stride_kfd,
                mask=q_mask, other=0.0,
            ).to(tl.float32)

        # ---- Score = dot(Q, K_t) * scale_factor ----
        score = tl.sum(q * k_val) * scale_factor

        # ====== Load V_t ======
        if t < T_v_quant:
            # V scale/mn: (B, nh_kv, T_v_quant, D//group_size)
            sm_base_v = (
                pid_b * stride_vsb
                + pid_h_kv * stride_vsh
                + t * stride_vst
            )
            d_group = offs_d // group_size
            scale_v = tl.load(
                v_scale_ptr + sm_base_v + d_group * stride_vsg,
                mask=q_mask, other=0.0,
            )
            mn_v = tl.load(
                v_mn_ptr + sm_base_v + d_group * stride_vmg,
                mask=q_mask, other=0.0,
            )

            d_packed = offs_d // feat_per_int
            d_bit = (offs_d % feat_per_int) * bits
            v_base = (
                pid_b * stride_vcb
                + pid_h_kv * stride_vch
                + t * stride_vct
            )
            v_code = tl.load(
                v_code_ptr + v_base + d_packed * stride_vcp,
                mask=q_mask, other=0,
            )
            v_val = (
                ((v_code >> d_bit) & num).to(tl.float32) * scale_v + mn_v
            )
        else:
            t_v_full = t - T_v_quant
            v_base = (
                pid_b * stride_vfb
                + pid_h_kv * stride_vfh
                + t_v_full * stride_vft
            )
            v_val = tl.load(
                v_full_ptr + v_base + offs_d * stride_vfd,
                mask=q_mask, other=0.0,
            ).to(tl.float32)

        # ---- Online softmax update ----
        m_new = tl.maximum(m_i, score)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(score - m_new)
        l_i = l_i * alpha + beta
        acc = acc * alpha + beta * v_val
        m_i = m_new

    # ---- Store partial results ----
    partial_base = pid_b * stride_pob + pid_h * stride_poh + pid_tile * stride_pot
    tl.store(partial_o_ptr + partial_base + offs_d * stride_pod, acc, mask=q_mask)
    base_ml = pid_b * stride_pmb + pid_h * stride_pmh + pid_tile * stride_pmt
    tl.store(partial_m_ptr + base_ml, m_i)
    tl.store(partial_l_ptr + base_ml, l_i)


@triton.jit
def kivi_decode_reduce_kernel(
    partial_o_ptr, partial_m_ptr, partial_l_ptr,
    out_ptr,
    B, nh, num_tiles, D,
    stride_pob, stride_poh, stride_pot, stride_pod,
    stride_pmb, stride_pmh, stride_pmt,
    stride_ob, stride_oh, stride_od,
    BLOCK_D: tl.constexpr,
):
    """Reduce partial attention results across tiles."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_d < D

    m_global = -float("inf")
    l_global = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    base_o = pid_b * stride_pob + pid_h * stride_poh
    base_ml = pid_b * stride_pmb + pid_h * stride_pmh
    for tile in range(num_tiles):
        m_tile = tl.load(partial_m_ptr + base_ml + tile * stride_pmt)
        l_tile = tl.load(partial_l_ptr + base_ml + tile * stride_pmt)
        o_tile = tl.load(
            partial_o_ptr + base_o + tile * stride_pot + offs_d * stride_pod,
            mask=q_mask, other=0.0,
        )

        m_new = tl.maximum(m_global, m_tile)
        alpha = tl.exp(m_global - m_new)
        beta = tl.exp(m_tile - m_new)
        l_global = l_global * alpha + l_tile * beta
        acc = acc * alpha + o_tile * beta
        m_global = m_new

    # Normalize and store
    acc = acc / l_global
    tl.store(
        out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od,
        acc, mask=q_mask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_empty_like(t, shape):
    if t.numel() == 0:
        return torch.empty(shape, device=t.device, dtype=t.dtype)
    return t


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def kivi_fused_decode_attention(
    q: torch.Tensor,
    k_code: torch.Tensor,
    k_scale: torch.Tensor,
    k_mn: torch.Tensor,
    k_full: torch.Tensor,
    v_code: torch.Tensor,
    v_scale: torch.Tensor,
    v_mn: torch.Tensor,
    v_full: torch.Tensor,
    scale: float,
    group_size: int,
    k_bits: int,
    v_bits: int,
) -> torch.Tensor:
    """Fused KIVI decode attention.

    Args:
        q:          (B, nh, 1, D)  query tensor
        k_code:     (B, nh_kv, T_k//feat_per_int, D) int32
        k_scale:    (B, nh_kv, T_k//group_size, D)
        k_mn:       same shape as k_scale
        k_full:     (B, nh_kv, T_k_full, D)
        v_code:     (B, nh_kv, T_v, D//feat_per_int) int32
        v_scale:    (B, nh_kv, T_v, D//group_size)
        v_mn:       same shape as v_scale
        v_full:     (B, nh_kv, T_v_full, D)
        scale:      softmax scale factor (typically 1/sqrt(D))
        group_size: quantization group size
        k_bits:     2 or 4
        v_bits:     2 or 4

    Returns:
        o: (B, nh, 1, D) attention output
    """
    assert q.dim() == 4 and q.shape[2] == 1
    B, nh, _, D = q.shape
    nh_kv = k_code.shape[1]
    assert nh % nh_kv == 0

    # Flatten M=1
    q_3d = q.squeeze(2)  # (B, nh, D)

    # k_code layout: (B, nh_kv, T_k_packed, D) where T_k_packed = T_k_quant // feat_per_int
    T_k_quant = k_code.shape[2] * (32 // k_bits)
    T_v_quant = v_code.shape[2] if v_code.numel() > 0 else 0
    T_k_full = k_full.shape[2] if k_full.numel() > 0 else 0
    T_v_full = v_full.shape[2] if v_full.numel() > 0 else 0
    total_len = T_k_quant + T_k_full
    # Sanity check: total_len should be the same for K and V
    assert total_len == T_v_quant + T_v_full, \
        f"Length mismatch: K={T_k_quant}+{T_k_full}={total_len}, V={T_v_quant}+{T_v_full}={T_v_quant+T_v_full}"

    if total_len == 0:
        return torch.zeros_like(q)

    # If only full-precision residual exists, fall back to simple matmul
    if T_k_quant == 0 and T_v_quant == 0:
        n_rep = nh // nh_kv
        if n_rep > 1 and k_full.numel() > 0:
            k_full = k_full.repeat_interleave(n_rep, dim=1)
            v_full = v_full.repeat_interleave(n_rep, dim=1)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v_full)
        return out

    # ---- Kernel config ----
    # For now use fixed block sizes; auto-tune can be added later.
    BLOCK_D = 128
    BLOCK_T = 64

    if D > BLOCK_D:
        # Fallback if head dim too large for fixed tile
        n_rep = nh // nh_kv
        if n_rep > 1:
            if k_full.numel() > 0:
                k_full = k_full.repeat_interleave(n_rep, dim=1)
            if v_full.numel() > 0:
                v_full = v_full.repeat_interleave(n_rep, dim=1)

        # k_code is in internal layout (B, nh, T_packed, D); matmul wrapper expects (B, nh, D, T_packed)
        k_code_t = k_code.transpose(2, 3).contiguous()
        k_scale_t = k_scale.transpose(2, 3).contiguous()
        k_mn_t = k_mn.transpose(2, 3).contiguous()
        att_qk = kivi_bmm_fA_qB_outer(group_size, q, k_code_t, k_scale_t, k_mn_t, k_bits)
        if k_full.numel() > 0:
            att_qk = torch.cat(
                [att_qk, torch.matmul(q, k_full.transpose(-2, -1))], dim=-1
            )
        attn_weights = torch.softmax(att_qk * scale, dim=-1, dtype=torch.float32).to(q.dtype)

        v_full_len = v_full.shape[2] if v_full.numel() > 0 else 0
        if v_code.numel() > 0:
            out = kivi_bmm_fA_qB_outer(
                group_size, attn_weights[:, :, :, :-v_full_len], v_code, v_scale, v_mn, v_bits
            )
            if v_full.numel() > 0:
                out += torch.matmul(attn_weights[:, :, :, -v_full_len:], v_full)
        else:
            out = torch.matmul(attn_weights, v_full)
        return out

    num_tiles = triton.cdiv(total_len, BLOCK_T)

    # Temporary buffers for partial results
    partial_o = torch.empty(B, nh, num_tiles, D, device=q.device, dtype=torch.float32)
    partial_m = torch.empty(B, nh, num_tiles, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(B, nh, num_tiles, device=q.device, dtype=torch.float32)

    # K full might be empty
    if k_full.numel() == 0:
        k_full = torch.empty(B, nh_kv, 0, D, device=q.device, dtype=q.dtype)
    if v_full.numel() == 0:
        v_full = torch.empty(B, nh_kv, 0, D, device=q.device, dtype=q.dtype)

    grid_fwd = (B, nh, num_tiles)
    kivi_decode_partial_kernel[grid_fwd](
        q_3d,
        k_code, k_scale, k_mn, k_full,
        v_code, v_scale, v_mn, v_full,
        partial_o, partial_m, partial_l,
        scale,
        B, nh, nh_kv, D, T_k_quant, T_v_quant, total_len,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_code.stride(0), k_code.stride(1), k_code.stride(2), k_code.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_scale.stride(3),
        k_mn.stride(0), k_mn.stride(1), k_mn.stride(2), k_mn.stride(3),
        k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
        v_code.stride(0), v_code.stride(1), v_code.stride(2), v_code.stride(3),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2), v_scale.stride(3),
        v_mn.stride(0), v_mn.stride(1), v_mn.stride(2), v_mn.stride(3),
        v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        bits=k_bits,
        group_size=group_size,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )

    out = torch.empty(B, nh, D, device=q.device, dtype=torch.float32)
    grid_reduce = (B, nh)
    kivi_decode_reduce_kernel[grid_reduce](
        partial_o, partial_m, partial_l,
        out,
        B, nh, num_tiles, D,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_D=BLOCK_D,
    )

    out = out.unsqueeze(2).to(q.dtype)  # (B, nh, 1, D)
    return out
