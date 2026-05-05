import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.awq_gemm import awq_gemm_forward


class WQLinear_W(nn.Module):
    """Quantized linear with weight in original W layout (out, in).
    Kernel computes X @ W^T with on-the-fly transpose.
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
        layout='W',
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
        assert self.layout == 'W', f"WQLinear_W expects layout='W', got {self.layout}"
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (out_features, in_features // self.group_size // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features, in_features // self.group_size),
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
                out_features,
                in_features // self.group_size,
                dtype=torch.float16,
                device=dev,
            ),
        )

        self.shifts = torch.arange(0, 32, self.w_bit, dtype=torch.int32, device=dev)

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
        layout='W',
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

        awq_linear.scales = scales.clone().half()
        if getattr(linear, 'bias', None) is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit

        intweight = []
        for idx in range(awq_linear.out_features):
            # shape = (1, in_features)
            row_weight = linear.weight.data[idx : idx + 1, :]

            # shape = (1, group_size)
            row_scales = scales[idx : idx + 1, :]
            row_scales_zeros = scale_zeros[idx : idx + 1, :]

            # shape = (1, in_features)
            row_scales_exp = row_scales.repeat_interleave(group_size, dim=1)
            row_scale_zeros_exp = row_scales_zeros.repeat_interleave(group_size, dim=1)

            row_intweight = torch.round(
                (row_weight + row_scale_zeros_exp) / row_scales_exp
            ).to(torch.int)

            row_intweight = torch.clamp(row_intweight, 0, 2**awq_linear.w_bit - 1)
            intweight.append(row_intweight)

        intweight = torch.cat(intweight, dim=0).contiguous().to(dtype=torch.int32)

        qweight = torch.zeros(
            (awq_linear.out_features, awq_linear.in_features // pack_num),
            dtype=torch.int32,
            device=intweight.device,
        )

        if awq_linear.pack_order == "sequential":
            order_map = [0, 1, 2, 3, 4, 5, 6, 7]
        elif awq_linear.pack_order == "swizzled":
            # order_map = [0, 4, 1, 5, 2, 6, 3, 7]
            raise NotImplementedError("swizzled pack_order is not implemented")
        else:
            raise ValueError(f"Unknown pack_order: {awq_linear.pack_order}")

        for col in range(intweight.shape[1] // pack_num):
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        qzeros = torch.zeros(
            (
                awq_linear.out_features,
                awq_linear.in_features // awq_linear.group_size // pack_num,
            ),
            dtype=torch.int32,
            device=zeros.device,
        )

        zeros_int = zeros.to(torch.int32)
        for col in range(zeros_int.shape[1] // pack_num):
            for i in range(pack_num):
                qzero_col = zeros_int[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        unpack_zeros = (awq_linear.qzeros.unsqueeze(-1) >> awq_linear.shifts) & 0xF
        awq_linear.unpack_zeros = unpack_zeros.reshape(
            awq_linear.out_features, awq_linear.in_features // awq_linear.group_size
        ).half()

        return awq_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if 'triton' in self.backend:
            return awq_gemm_forward(
                x,
                self.qweight,
                self.scales,
                self.unpack_zeros,
                self.group_size,
                32 // self.w_bit,
                self.bias,
                self.backend,
            )
        else:
            pack_num = 32 // self.w_bit
            shifts = self.shifts

            intweight = (self.qweight.unsqueeze(-1) >> shifts) & 0xF

            intweight = intweight.reshape(
                self.out_features, self.in_features // self.group_size, self.group_size
            )

            weight = (
                intweight.float() - self.unpack_zeros.unsqueeze(-1).float()
            ) * self.scales.unsqueeze(-1).float()

            weight = weight.reshape(self.out_features, self.in_features)

            weight = weight.to(x.dtype)
            bias = self.bias.to(x.device).to(x.dtype) if self.bias is not None else None
            return F.linear(x, weight, bias)


# Backward-compatible alias
WQLinear_GEMM = WQLinear_W
