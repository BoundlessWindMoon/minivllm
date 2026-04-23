import torch.nn as nn
from dataclasses import dataclass

from layers.activation import SiluAndMul
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.cohere.modeling_cohere import CohereLayerNorm
from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation

allowed_norms = [
    nn.LayerNorm,
    LlamaRMSNorm,
    GemmaRMSNorm,
    Gemma2RMSNorm,
    CohereLayerNorm,
]

allowed_act_fns = [
    nn.GELU,
    BloomGelu,
    NewGELUActivation,
    PytorchGELUTanh,
    GELUActivation,
    SiluAndMul,
]


@dataclass
class QuantConfig:
    quant_method: str = "AWQ"
    quant_bits: int = 4
    quant_targets: list = None
    group_size: int = 128
    has_zero_point: bool = True

    def to_dict(self):
        return self.__dict__
