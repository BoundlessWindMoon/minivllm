"""AWQ quantized linear with Wt layout fused kernel."""

import torch
import torch.nn as nn

from kernels.awq_gemm_wt import awq_gemm_forward_wt
from kernels.awq_gemm_wt_fused import awq_gemm_forward_wt_fused


class WQLinear_Wt(nn.Module):
    """Quantized linear layer with pre-transposed W^T layout.

    All quantized buffers are stored in W^T layout:
      - qweight:     (in_features, out_features // num_pack)
      - qzeros:      (in_features // group_size, out_features // num_pack)
      - scales:      (in_features // group_size, out_features)
      - unpack_zeros:(in_features // group_size, out_features)
      - zero_scales: (in_features // group_size, out_features)   # unpack_zeros * scales

    Triton path:       awq_gemm_forward_wt (no transpose inside kernel).
    Triton fused path: awq_gemm_forward_wt_fused (hoisted scale + FMA dequant).
    Naive fallback:    uses globally lazied dequantized_weight_t (in, out).
    """

    def __init__(
        self,
        w_bit,
        group_size,
        in_features,
        out_features,
        bias,
        dev,
        training=False,
        backend='gemm',
        layout='Wt',
        pack_order='sequential',
    ):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training
        self.backend = backend
        self.layout = layout
        self.pack_order = pack_order
        assert (
            self.layout == 'Wt'
        ), f"WQLinear_Wt expects layout='Wt', got {self.layout}"
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        pack_num = 32 // self.w_bit

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // pack_num),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // pack_num),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

        self.register_buffer(
            "unpack_zeros",
            torch.zeros(
                in_features // self.group_size,
                out_features,
                dtype=torch.float16,
                device=dev,
            ),
        )
        self.register_buffer(
            "zero_scales",
            torch.zeros(
                in_features // self.group_size,
                out_features,
                dtype=torch.float16,
                device=dev,
            ),
        )

        self.shifts = torch.arange(0, 32, self.w_bit, dtype=torch.int32, device=dev)

        self.register_buffer("dequantized_weight_t", None)

    def _post_materialize_fixup(self, device):
        # WHY: self.shifts is a plain attribute (not a buffer); meta-skeleton
        # leaves it on meta and must be rebuilt on the compute device.
        self.shifts = torch.arange(
            0, 32, self.w_bit, dtype=torch.int32, device=device
        )

    @classmethod
    def from_linear(
        cls,
        linear,
        w_bit,
        group_size,
        init_only=False,
        scales=None,
        zeros=None,
        backend='gemm',
        layout='Wt',
        pack_order='sequential',
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.weight.shape[1],
            linear.weight.shape[0],
            bias=getattr(linear, 'bias', None) is not None,
            dev=linear.weight.device,
            backend=backend,
            layout=layout,
            pack_order=pack_order,
        )
        if init_only:
            return awq_linear

        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        awq_linear.scales = scales.t().half()
        if getattr(linear, 'bias', None) is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        # Batch-process all rows at once (was row-wise loop, O(N) overhead for large N)
        scales_exp = scales.repeat_interleave(group_size, dim=1)
        scale_zeros_exp = scale_zeros.repeat_interleave(group_size, dim=1)
        intweight = torch.round(
            (linear.weight.data + scale_zeros_exp) / scales_exp
        ).to(torch.int)
        intweight = torch.clamp(intweight, 0, 2**awq_linear.w_bit - 1)
        intweight = intweight.contiguous().to(dtype=torch.int32)

        intweight_t = intweight.t()

        if awq_linear.pack_order == "sequential":
            order_map = [0, 1, 2, 3, 4, 5, 6, 7]
        elif awq_linear.pack_order == "swizzled":
            order_map = [0, 4, 1, 5, 2, 6, 3, 7]
        else:
            raise ValueError(f"Unknown pack_order: {awq_linear.pack_order}")

        qweight_t = torch.zeros(
            (awq_linear.in_features, awq_linear.out_features // pack_num),
            dtype=torch.int32,
            device=intweight.device,
        )
        for col in range(intweight_t.shape[1] // pack_num):
            for i in range(pack_num):
                qweight_col = intweight_t[:, col * pack_num + order_map[i]]
                qweight_t[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight_t

        qzeros = torch.zeros(
            (
                awq_linear.in_features // awq_linear.group_size,
                awq_linear.out_features // pack_num,
            ),
            dtype=torch.int32,
            device=zeros.device,
        )
        zeros_int = zeros.to(torch.int32).t()
        for col in range(zeros_int.shape[1] // pack_num):
            for i in range(pack_num):
                qzero_col = zeros_int[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        unpack_zeros = (awq_linear.qzeros.unsqueeze(-1) >> awq_linear.shifts) & 0xF
        awq_linear.unpack_zeros = unpack_zeros.reshape(
            awq_linear.in_features // awq_linear.group_size, awq_linear.out_features
        ).half()
        awq_linear.zero_scales = (awq_linear.unpack_zeros * awq_linear.scales).half()

        return awq_linear

    @torch.compile
    def _dequantize_to_weight_t(self, dtype):
        intweight = (self.qweight.unsqueeze(-1) >> self.shifts) & 0xF

        in_f, out_f = self.in_features, self.out_features
        gs = self.group_size

        intweight_grouped = intweight.reshape(in_f // gs, gs, out_f)

        scales_grouped = self.scales.unsqueeze(1)
        zeros_grouped = self.unpack_zeros.unsqueeze(1)

        weight_grouped = (
            intweight_grouped.float() - zeros_grouped.float()
        ) * scales_grouped.float()

        weight = weight_grouped.reshape(in_f, out_f)

        return weight.to(dtype).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if 'triton_wt_fused' in self.backend:
            return awq_gemm_forward_wt_fused(
                x,
                self.qweight,
                self.scales,
                self.zero_scales,
                self.group_size,
                32 // self.w_bit,
                self.bias,
            )
        elif 'triton_wt' in self.backend:
            return awq_gemm_forward_wt(
                x,
                self.qweight,
                self.scales,
                self.unpack_zeros,
                self.group_size,
                32 // self.w_bit,
                self.bias,
            )
        else:
            if self.dequantized_weight_t is None:
                self.dequantized_weight_t = self._dequantize_to_weight_t(x.dtype)

            if self.bias is not None:
                bias = self.bias.to(x.device).to(x.dtype)
                return torch.matmul(x, self.dequantized_weight_t) + bias
            return torch.matmul(x, self.dequantized_weight_t)
