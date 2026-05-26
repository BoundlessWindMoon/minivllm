"""Shared MLP implementation used by all model families."""

from torch import nn

from layers.linear import MergedColumnParallelLinear, RowParallelLinear
from layers.activation import SiluAndMul


class SiluMLP(nn.Module):
    """Standard SwiGLU MLP: gate_up_proj -> SiLU+Mul -> down_proj."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=bias,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x) -> None:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)
