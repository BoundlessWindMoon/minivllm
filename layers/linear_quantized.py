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
        # 1. 搬运原有属性
        self.weight = original_layer.weight
        if hasattr(original_layer.weight, 'weight_loader'):
            self.weight.weight_loader = original_layer.weight.weight_loader
        # 修复：无论 bias 是否为 None，都要赋值给 self.bias
        self.bias = original_layer.bias
        # 如果 bias 存在，同样要把它的 weight_loader 搬过来
        if self.bias is not None and hasattr(original_layer.bias, 'weight_loader'):
            self.bias.weight_loader = original_layer.bias.weight_loader
        # 保留 TP 相关属性，确保后续加载权重不出错
        self.tp_dim = getattr(original_layer, 'tp_dim', None)
        self.tp_rank = getattr(original_layer, 'tp_rank', 0)
        self.tp_size = getattr(original_layer, 'tp_size', 1)
        # 2. 注入量化必要的 Tensor (先用全 1/0 占位，跑通流程)
        out_features = original_layer.weight.shape[0]
        in_features = original_layer.weight.shape[1]
        # 假设按 group_size 切分 scale 和 zp
        self.group_size = group_size
        scale_zeros_shape = (out_features, in_features // group_size)
        # 使用 register_buffer 而非 nn.Parameter，因为这些不需要梯度更新
        self.register_buffer(
            'q_scale', torch.ones(scale_zeros_shape, dtype=torch.float16)
        )
        self.register_buffer(
            'q_zero_point', torch.zeros(scale_zeros_shape, dtype=torch.int8)
        )
        # 也可以提前把 q_weight 的坑占上，虽然现在还不填数据
        self.register_buffer(
            'q_weight', torch.zeros_like(original_layer.weight, dtype=torch.int8)
        )

    def forward(self, x):
        # 跑通流程阶段：依然使用原始的 fp16/fp32 权重计算
        # 后续优化时，会在这里替换为量化反量化 kernel 或调用 bitsandbytes
        return F.linear(x, self.weight, self.bias)
