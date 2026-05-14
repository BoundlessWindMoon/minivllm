"""Triton-accelerated SiLU-and-Mul activation."""

import torch
from torch import nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['N'],
)
@triton.jit
def _silu_and_mul_kernel(
    X_ptr,
    Out_ptr,
    Scales_ptr,
    N,
    Half_N,
    stride_x_m,
    stride_out_m,
    HAS_SCALES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    x_ptr = X_ptr + row * stride_x_m
    out_ptr = Out_ptr + row * stride_out_m

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < Half_N

    x_vals = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y_vals = tl.load(x_ptr + Half_N + cols, mask=mask, other=0.0).to(tl.float32)

    silu_vals = x_vals * tl.sigmoid(x_vals)

    out_vals = silu_vals * y_vals

    if HAS_SCALES:
        scales_vals = tl.load(Scales_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        out_vals = out_vals / scales_vals

    tl.store(out_ptr + cols, out_vals, mask=mask)


def triton_silu_and_mul(
    x: torch.Tensor, scales: torch.Tensor | None = None
) -> torch.Tensor:
    M, N = x.shape
    Half_N = N // 2

    out = torch.empty((M, Half_N), dtype=x.dtype, device=x.device)

    grid = (M,)

    BLOCK_SIZE = triton.next_power_of_2(Half_N)

    HAS_SCALES = scales is not None

    if scales is None:
        scales_ptr = torch.empty(1, device=x.device)
    else:
        scales_ptr = scales
        assert scales.shape == (
            Half_N,
        ), f"Scales shape {scales.shape} mismatch with Half_N={Half_N}"

    _silu_and_mul_kernel[grid](
        x,
        out,
        scales_ptr,
        N,
        Half_N,
        x.stride(0),
        out.stride(0),
        HAS_SCALES=HAS_SCALES,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class SiluAndMul(nn.Module):
    def __init__(self, scales=None):
        super().__init__()
        if scales is not None:
            if not isinstance(scales, torch.Tensor):
                scales = torch.tensor(scales, dtype=torch.float32)
            self.register_buffer("scales", scales)
        else:
            self.scales = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        out_flat = triton_silu_and_mul(x_flat, self.scales)
        return out_flat.reshape(*orig_shape, out_flat.shape[-1])
