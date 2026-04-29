import os
import json
import shutil
import torch
from safetensors.torch import save_file, load_file
from typing import List

from layers.quanted_linear import WQLinear_GEMM
from layers.quanted_linear_cached import WQLinear_GEMM_Cached
from utils.model_utils import set_op_by_name
from utils.scale_utils import ScaledActivation
from utils.logger import logger


def _get_quantized_layer_names(model) -> List[str]:
    return [
        name for name, m in model.named_modules() if isinstance(m, WQLinear_GEMM_Cached)
    ]


def _get_scaled_activation_info(model) -> dict:
    return {
        name: list(m.scales.shape)
        for name, m in model.named_modules()
        if isinstance(m, ScaledActivation)
    }


def _ensure_contiguous(state_dict: dict) -> dict:
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
            state_dict[key] = tensor.contiguous()
    return state_dict


def save_quantized_model(model, save_path: str, quant_config, original_model_path: str):
    os.makedirs(save_path, exist_ok=True)

    was_tied = False
    if getattr(model.config, "tie_word_embeddings", False):
        was_tied = True
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

    state_dict = _ensure_contiguous(model.state_dict())
    save_file(state_dict, os.path.join(save_path, "model.safetensors"))

    quant_info = {
        "quant_method": quant_config.quant_method,
        "quant_bits": quant_config.quant_bits,
        "group_size": quant_config.group_size,
        "has_zero_point": quant_config.has_zero_point,
        "quantized_layers": _get_quantized_layer_names(model),
        "scaled_activations": _get_scaled_activation_info(model),
        "tie_word_embeddings": was_tied,
    }

    with open(os.path.join(save_path, "quant_config.json"), "w") as f:
        json.dump(quant_info, f, indent=2)

    src_config = os.path.join(original_model_path, "config.json")
    dst_config = os.path.join(save_path, "config.json")
    if os.path.exists(src_config):
        shutil.copy2(src_config, dst_config)


def _replace_layer_with_quantized(
    model, layer_name: str, quant_bits: int, group_size: int, backend: str
):
    linear_layer = model.get_submodule(layer_name)
    q_linear = WQLinear_GEMM_Cached.from_linear(
        linear=linear_layer,
        w_bit=quant_bits,
        group_size=group_size,
        init_only=True,
        backend=backend,
    )
    set_op_by_name(model, layer_name, q_linear)


def _restore_scaled_activation(model, act_name: str, shape: list):
    parent_name, attr_name = act_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    original_act = getattr(parent, attr_name)

    if isinstance(original_act, ScaledActivation):
        return

    device = next(model.parameters()).device
    wrapped = ScaledActivation(original_act, torch.ones(shape, device=device))
    setattr(parent, attr_name, wrapped)


def prepare_model_for_quantized_load(model, quant_info: dict, backend: str):
    for layer_name in quant_info.get("quantized_layers", []):
        _replace_layer_with_quantized(
            model,
            layer_name,
            quant_info["quant_bits"],
            quant_info["group_size"],
            backend,
        )

    scaled_info = quant_info.get("scaled_activations", {})
    if isinstance(scaled_info, dict):
        for act_name, shape in scaled_info.items():
            _restore_scaled_activation(model, act_name, shape)
    elif isinstance(scaled_info, list):
        for act_name in scaled_info:
            _restore_scaled_activation(model, act_name, [1])


def load_quantized_weights(model, model_path: str, quant_info: dict):
    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        raise RuntimeError(f"Missing keys when loading quantized model: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading quantized model: {unexpected}")

    if quant_info.get("tie_word_embeddings"):
        model.lm_head.weight = model.model.embed_tokens.weight

    return model


def is_quantized_model(model_path: str) -> bool:
    quant_config_path = os.path.join(model_path, "quant_config.json")
    if not os.path.exists(quant_config_path):
        return False
    try:
        with open(quant_config_path, "r") as f:
            info = json.load(f)
        return isinstance(info.get("quantized_layers"), list)
    except (json.JSONDecodeError, OSError):
        return False
