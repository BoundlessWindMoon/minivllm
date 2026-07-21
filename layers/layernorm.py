"""RMSNorm / LayerNorm implementations."""

import torch
import torch.nn.functional as F
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
    key=["N"],
)
@triton.jit
def _rms_norm_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    N,
    eps,
    stride_x_m,
    stride_x_n,
    stride_out_m,
    stride_out_n,
    BLOCK_SIZE: tl.constexpr,
):

    row = tl.program_id(0)

    x_ptr = X_ptr + row * stride_x_m
    out_ptr = Out_ptr + row * stride_out_m

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N

    rrms = 1.0 / tl.sqrt(var + eps)

    y = x * rrms * w

    tl.store(out_ptr + cols * stride_out_n, y, mask=mask)


def triton_rms_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    M, N = x.shape
    orig_dtype = x.dtype

    out = torch.empty_like(x)

    assert weight.shape == (N,), f"Weight shape {weight.shape} mismatch with N={N}"

    grid = (M,)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _rms_norm_kernel[grid](
        x,
        weight,
        out,
        N,
        eps,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        centered: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.centered = centered
        self.weight = nn.Parameter(
            torch.zeros(hidden_size) if centered else torch.ones(hidden_size)
        )

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        if self.centered:
            x = x.mul_(1.0 + self.weight.float())
        else:
            x = x.mul_(self.weight.float())
        return x.to(orig_dtype)

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        if self.centered:
            x = x.mul_(1.0 + self.weight.float())
        else:
            x = x.mul_(self.weight.float())
        return x.to(orig_dtype), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if self.centered:
                # Triton kernel does not support centered mode; fallback to PyTorch.
                original_shape = x.shape
                return self.rms_forward(x.reshape(-1, x.shape[-1])).reshape(
                    original_shape
                )
            original_shape = x.shape
            return triton_rms_forward(
                x.reshape(-1, x.shape[-1]), self.weight, self.eps
            ).reshape(original_shape)
        else:
            return self.add_rms_forward(x, residual)


class RMSNormGated(nn.Module):
    """RMSNorm with an additional SiLU gate (used by Qwen3.5 linear attention)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.compile
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
