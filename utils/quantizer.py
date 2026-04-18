import torch
import torch.nn as nn

from utils.logger import logger
from dataclasses import dataclass
from tqdm import tqdm
from layers.attention import Attention
from layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


@dataclass
class QuantConfig:
    quant_method: str = "AWQ"
    quant_bits: int = 8
    quant_targets: list = None
    group_size: int = 128
    has_zero_point: bool = True

    def to_dict(self):
        return self.__dict__


class Quantizer:
    def __init__(
        self,
        model,
        tokenizer,
        quant_config: QuantConfig,
        device="cuda:0",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.quant_config = quant_config
        self.device = device

    def _calibrate(self):
        if self.quant_config.quant_method == "AWQ":
            pass
        pass

    def _get_target_layers(self):
        target_layers = []
        for target in self.quant_config.quant_targets:
            if target == "MLP":
                target_layers.append(MergedColumnParallelLinear)
                target_layers.append(RowParallelLinear)
                pass
            elif target == "ATTENTION":
                target_layers.append(QKVParallelLinear)
                target_layers.append(RowParallelLinear)
            else:
                logger.error(f"Unknown quantization target: {target}")
                raise ValueError("quantization target not supported")
        return tuple(dict.fromkeys(target_layers))

    def _quantize_layer(self, layer):
        from layers.linear_quantized import QuantizedLinearWrapper

        quantized_layer = QuantizedLinearWrapper(
            original_layer=layer, group_size=self.quant_config.group_size
        )
        return quantized_layer

    def _set_layer_by_name(self, name, layer):
        # parent_path be like:  ['model', 'layers', '0', 'self_attn']
        # child_attr be like: qkv_proj
        *parent_path, child_attr = name.split(".")

        parent = self.model
        for attr in parent_path:
            parent = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
        setattr(parent, child_attr, layer)

    def _quantize_and_replace(self):
        target_layers = self._get_target_layers()
        for name, layer in tqdm(list(self.model.named_modules()), desc="Quantizing"):
            if isinstance(layer, target_layers):
                quantized_layer = self._quantize_layer(layer)
                self._set_layer_by_name(name, quantized_layer)

    def _save_quantized_model(self):
        pass

    def run(self):
        self._calibrate()
        self._quantize_and_replace()
        # quantized_model = self._save_quantized_model()

        return self.model
