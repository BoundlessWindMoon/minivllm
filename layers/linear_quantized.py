import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinearWrapper(nn.Module):
    """
    包装原有的 Linear 层：
    1. 保留原有的 weight, bias 以及至关重要的 weight_loader
    2. 注入量化所需的 scale 和 zero_point (使用 register_buffer)
    3. forward 暂时保持原样（使用 fp16 计算），确保流程跑通
    """

    def __init__(self, original_layer: nn.Module, group_size: int = 128):
        super().__init__()
        self.weight = original_layer.weight
        if hasattr(original_layer.weight, 'weight_loader'):
            self.weight.weight_loader = original_layer.weight.weight_loader

        self.bias = original_layer.bias

        if self.bias is not None and hasattr(original_layer.bias, 'weight_loader'):
            self.bias.weight_loader = original_layer.bias.weight_loader

        self.tp_dim = getattr(original_layer, 'tp_dim', None)
        self.tp_rank = getattr(original_layer, 'tp_rank', 0)
        self.tp_size = getattr(original_layer, 'tp_size', 1)

        out_features = original_layer.weight.shape[0]
        in_features = original_layer.weight.shape[1]

        self.group_size = group_size
        scale_zeros_shape = (out_features, in_features // group_size)

        self.register_buffer(
            'q_scale', torch.ones(scale_zeros_shape, dtype=torch.float16)
        )
        self.register_buffer(
            'q_zero_point', torch.zeros(scale_zeros_shape, dtype=torch.int8)
        )

        self.register_buffer(
            'q_weight', torch.zeros_like(original_layer.weight, dtype=torch.int8)
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
