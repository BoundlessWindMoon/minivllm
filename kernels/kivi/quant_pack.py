"""KIVI quantization / packing kernels (PyTorch + optional Triton).

Ported from KIVI/quant/new_pack.py for mini-vllm integration.
Supports 2-bit and 4-bit asymmetric min-max quantization with bit-packing.
"""

import numpy as np
import torch
import torch.nn.functional as F


def quantize_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
    """Quantize K cache per-channel (along seq_len) and pack along seq_len.

    Args:
        k: (B, nh, T, D)  — will be quantized along T
        group_size: group size along the quantization dimension
        bits: 2 or 4

    Returns:
        code:   (B, nh, T // feat_per_int, D) int32  (packed along T)
        scale:  (B, nh, T // group_size, D)
        mn:     (B, nh, T // group_size, D)
    """
    assert len(k.shape) == 4
    assert bits in (2, 4)
    B, nh, T, D = k.shape
    assert T % group_size == 0, f"T={T} must be divisible by group_size={group_size}"
    num_groups = T // group_size
    new_shape = (B, nh, num_groups, group_size, D)

    max_int = 2 ** bits - 1
    data = k.view(new_shape)
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale = (mx - mn) / max_int

    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(B, nh, T, D)
    code = pack_tensor(data, bits, pack_dim=2)
    return code, scale.squeeze(-2), mn.squeeze(-2)


def quantize_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
    """Quantize V cache per-token (along head_dim) and pack along head_dim.

    Args:
        v: (B, nh, T, D)  — will be quantized along D
        group_size: group size along D
        bits: 2 or 4

    Returns:
        code:   (B, nh, T, D // feat_per_int) int32  (packed along D)
        scale:  (B, nh, T, D // group_size)
        mn:     (B, nh, T, D // group_size)
    """
    assert len(v.shape) == 4
    assert bits in (2, 4)
    B, nh, T, D = v.shape
    assert D % group_size == 0, f"D={D} must be divisible by group_size={group_size}"
    num_groups = D // group_size
    new_shape = (B, nh, T, num_groups, group_size)

    max_int = 2 ** bits - 1
    data = v.view(new_shape)
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]
    scale = (mx - mn) / max_int

    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(B, nh, T, D)
    code = pack_tensor(data, bits, pack_dim=3)
    return code, scale.squeeze(-1), mn.squeeze(-1)


def unpack_and_dequant_kcache(
    k_code: torch.IntTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
    out_dtype: torch.dtype = torch.float16,
):
    """Unpack and dequantize K cache.

    Args:
        k_code: (B, nh, T_packed, D) int32
        scale:  (B, nh, T // group_size, D)
        mn:     (B, nh, T // group_size, D)
        out_dtype: dtype of returned tensor

    Returns:
        (B, nh, T, D) of *out_dtype*
    """
    assert bits in (2, 4)
    assert len(k_code.shape) == 4
    data = unpack_tensor(k_code, bits, pack_dim=2)
    shape = data.shape
    num_groups = shape[2] // group_size
    data = data.view(shape[:2] + (num_groups, group_size) + shape[3:])
    data = data.to(out_dtype)
    data = data * scale.unsqueeze(-2) + mn.unsqueeze(-2)
    return data.view(shape)


def unpack_and_dequant_vcache(
    v_code: torch.IntTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
    out_dtype: torch.dtype = torch.float16,
):
    """Unpack and dequantize V cache.

    Args:
        v_code: (B, nh, T, D_packed) int32
        scale:  (B, nh, T, D // group_size)
        mn:     (B, nh, T, D // group_size)
        out_dtype: dtype of returned tensor

    Returns:
        (B, nh, T, D) of *out_dtype*
    """
    assert bits in (2, 4)
    assert len(v_code.shape) == 4
    data = unpack_tensor(v_code, bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(shape[:3] + (num_groups, group_size))
    data = data.to(out_dtype)
    data = data * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return data.view(shape)


# ---------------------------------------------------------------------------
# Bit packing helpers
# ---------------------------------------------------------------------------

def pack_tensor(data: torch.Tensor, bits: int, pack_dim: int):
    """Pack int32 values (0..2**bits-1) into 32-bit integers.

    Args:
        data: int32 tensor, the dimension *pack_dim* must be divisible by
              32 // bits.
        bits: 2 or 4
        pack_dim: dimension along which to pack

    Returns:
        int32 tensor with *pack_dim* shrunk by feat_per_int.
    """
    shape = data.shape
    feat_per_int = 32 // bits
    assert bits in (2, 4), "Only 2, 4 bits supported"
    assert shape[pack_dim] % feat_per_int == 0, (
        f"dim {pack_dim} ({shape[pack_dim]}) must be divisible by {feat_per_int}"
    )

    new_shape = (
        shape[:pack_dim]
        + (shape[pack_dim] // feat_per_int,)
        + shape[pack_dim + 1 :]
    )
    code = torch.zeros(new_shape, dtype=torch.int32, device=data.device)

    unpacked_indices = [slice(None)] * len(data.shape)
    packed_indices = [slice(None)] * len(new_shape)

    row = 0
    i = 0
    while row < code.shape[pack_dim]:
        packed_indices[pack_dim] = row
        for j in range(i, i + feat_per_int):
            unpacked_indices[pack_dim] = j
            code[tuple(packed_indices)] |= data[tuple(unpacked_indices)] << (bits * (j - i))
        i += feat_per_int
        row += 1
    return code


def unpack_tensor(v_code: torch.Tensor, bits: int, pack_dim: int):
    """Unpack 32-bit integers into int8 values.

    Args:
        v_code: int32 tensor
        bits: 2 or 4
        pack_dim: dimension that was packed

    Returns:
        int8 tensor with *pack_dim* expanded by feat_per_int.
    """
    assert bits in (2, 4)
    shape = v_code.shape
    feat_per_int = 32 // bits
    new_shape = (
        shape[:pack_dim]
        + (shape[pack_dim] * feat_per_int,)
        + shape[pack_dim + 1 :]
    )
    unpacked = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)

    i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
    j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
    num = 0xFF >> (8 - bits)

    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i

    if pack_dim == 2:
        unpacked = ((v_code[tuple(packed_indices)] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
    elif pack_dim == 3:
        unpacked = ((v_code[tuple(packed_indices)] >> (j * bits)).to(torch.int16)) & num
    else:
        raise NotImplementedError(f"pack_dim={pack_dim} not supported")
    return unpacked
