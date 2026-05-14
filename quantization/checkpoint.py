"""Save/load quantized checkpoints and weight packing."""

import os
import json
import shutil
import torch
from safetensors.torch import save_file, load_file
from typing import List

from quantization.quantized_linear import WQLinear_W
from quantization.quantized_linear_wt import WQLinear_Wt
from quantization.module_ops import set_op_by_name
from quantization.scale import ScaledActivation
from utils.logger import logger


def _get_quantized_layer_names(model) -> List[str]:
    return [
        name
        for name, m in model.named_modules()
        if isinstance(m, (WQLinear_W, WQLinear_Wt))
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
        if not isinstance(model.lm_head, (WQLinear_W, WQLinear_Wt)):
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

    state_dict = _ensure_contiguous(model.state_dict())
    save_file(state_dict, os.path.join(save_path, "model.safetensors"))

    quant_info = {
        "quant_method": quant_config.quant_method,
        "quant_bits": quant_config.quant_bits,
        "group_size": quant_config.group_size,
        "has_zero_point": quant_config.has_zero_point,
        "layout": getattr(quant_config, 'layout', 'Wt'),
        "pack_order": getattr(quant_config, 'pack_order', 'sequential'),
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
    model,
    layer_name: str,
    quant_bits: int,
    group_size: int,
    backend: str,
    layout: str = 'Wt',
    pack_order: str = 'sequential',
):
    linear_layer = model.get_submodule(layer_name)
    if layout == 'W':
        q_cls = WQLinear_W
    elif layout == 'Wt':
        q_cls = WQLinear_Wt
    else:
        raise ValueError(f"Unknown layout: {layout}")
    q_linear = q_cls.from_linear(
        linear=linear_layer,
        w_bit=quant_bits,
        group_size=group_size,
        init_only=True,
        backend=backend,
        layout=layout,
        pack_order=pack_order,
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
    layout = quant_info.get("layout", "Wt")
    pack_order = quant_info.get("pack_order", "sequential")
    for layer_name in quant_info.get("quantized_layers", []):
        _replace_layer_with_quantized(
            model,
            layer_name,
            quant_info["quant_bits"],
            quant_info["group_size"],
            backend,
            layout=layout,
            pack_order=pack_order,
        )

    scaled_info = quant_info.get("scaled_activations", {})
    if isinstance(scaled_info, dict):
        for act_name, shape in scaled_info.items():
            _restore_scaled_activation(model, act_name, shape)
    elif isinstance(scaled_info, list):
        for act_name in scaled_info:
            _restore_scaled_activation(model, act_name, [1])


def _shard_paths(model_path: str) -> list[str]:
    """Resolve a list of safetensors files to load, in order.

    Supports both single-file (`model.safetensors`) and multi-shard
    (`model.safetensors.index.json` + `model-*-of-*.safetensors`) layouts.
    """
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {model_path}"
        )
    with open(index_path) as f:
        idx = json.load(f)
    shard_files = sorted(set(idx["weight_map"].values()))
    return [os.path.join(model_path, s) for s in shard_files]


def load_quantized_weights(
    model, model_path: str, quant_info: dict, expected_config=None
):
    shards = _shard_paths(model_path)
    # Load and apply shard-by-shard so peak host RAM is bounded by one shard
    # instead of the full state_dict.
    all_unexpected: list[str] = []
    loaded_keys: set[str] = set()
    for shard_path in shards:
        sd_shard = load_file(shard_path, device="cpu")
        _, unexpected = model.load_state_dict(sd_shard, strict=False)
        loaded_keys.update(sd_shard.keys())
        all_unexpected.extend(unexpected)
        del sd_shard

    expected_keys = set(model.state_dict().keys())
    missing = list(expected_keys - loaded_keys)

    zero_scales_missing = [k for k in missing if k.endswith(".zero_scales")]
    for key in zero_scales_missing:
        module = model.get_submodule(key.rsplit(".", 1)[0])
        with torch.no_grad():
            module.zero_scales.copy_((module.unpack_zeros * module.scales).half())
        missing.remove(key)

    if missing:
        raise RuntimeError(f"Missing keys when loading quantized model: {missing}")
    if all_unexpected:
        logger.warning(f"Unexpected keys when loading quantized model: {all_unexpected}")

    if expected_config is not None:
        saved_layout = quant_info.get("layout", "Wt")
        saved_pack = quant_info.get("pack_order", "sequential")
        expected_layout = getattr(expected_config, "layout", "Wt")
        expected_pack = getattr(expected_config, "pack_order", "sequential")
        if saved_layout != expected_layout:
            raise RuntimeError(
                f"Layout mismatch: model was saved with layout='{saved_layout}', "
                f"but config expects layout='{expected_layout}'"
            )
        if saved_pack != expected_pack:
            raise RuntimeError(
                f"Pack order mismatch: model was saved with pack_order='{saved_pack}', "
                f"but config expects pack_order='{expected_pack}'"
            )

    if quant_info.get("tie_word_embeddings"):
        quantized_layers = quant_info.get("quantized_layers", [])
        if "lm_head" not in quantized_layers:
            model.lm_head.weight = model.model.embed_tokens.weight
        else:
            logger.info(
                "lm_head is quantized; skipping tie_word_embeddings to keep independent quantized weight."
            )

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
