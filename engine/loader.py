"""Top-level model loader.

Handles: tokenizer load, quantized vs fp/bf16 branch, meta-skeleton
materialization with optional CPU offload, megakernel backend switch,
generation_config attachment.
"""

import os
import json

import torch
from transformers import AutoTokenizer, AutoConfig

from utils.logger import logger
from utils.config import GlobalConfig, resolve_data_path
from utils.model_loader import ModelLoader
from utils.cpu_offload import materialize_with_offload, apply_cpu_offload
from quantization.checkpoint import (
    prepare_model_for_quantized_load,
    load_quantized_weights,
)
from model.factory import create_base_model, create_megakernel_model


def _propagate_config_fields(config, cfg: GlobalConfig):
    """Propagate runtime preferences into the HF model config."""
    config.linear_attn_prefill_backend = getattr(
        cfg.inference, "linear_attn_prefill_backend", "torch"
    )
    config.linear_attn_decode_backend = getattr(
        cfg.inference, "linear_attn_decode_backend", "fla"
    )
    _kv_max = getattr(cfg.inference, "kv_cache_max_len", None)
    if _kv_max is not None:
        config.kv_cache_max_len = _kv_max
    config.preallocate_cache = cfg.inference.use_cuda_graph


def _load_weights(model, cfg: GlobalConfig, data_path: str):
    """Load model weights (quantized or fp/bf16)."""
    if cfg.inference.use_quantized_model:
        quant_config_path = os.path.join(data_path, "quant_config.json")
        with open(quant_config_path) as f:
            quant_info = json.load(f)
        prepare_model_for_quantized_load(model, quant_info, cfg.quant.backend)
        offload_paths = list(cfg.inference.cpu_offload_modules)
        materialize_with_offload(model, cfg.env.device, offload_paths)
        if offload_paths:
            apply_cpu_offload(model, offload_paths, cfg.env.device)
            logger.info(f"CPU offload enabled for: {offload_paths}")
            if cfg.inference.use_cuda_graph:
                logger.warning(
                    "CPU offload is incompatible with CUDA graph capture "
                    "(stream capture forbids host<->device copies and CPU "
                    "kernels); disabling use_cuda_graph for this run."
                )
                cfg.inference.use_cuda_graph = False
        load_quantized_weights(model, data_path, quant_info, expected_config=cfg.quant)
    else:
        loader = ModelLoader(data_path)
        model = loader.inject_data(model)
    return model


def load_model(cfg: GlobalConfig):
    """Load model and tokenizer from config.
    Returns:
        (model, tokenizer)
    """
    data_path = resolve_data_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)
    backend = cfg.inference.backend

    if cfg.inference.use_quantized_model and backend == "megakernel_cuda":
        raise RuntimeError(
            "Megakernel backend does not support quantized models. "
            "Please set inference.backend to 'default' when using a quantized model."
        )

    config_path = (
        data_path if cfg.inference.use_quantized_model else cfg.path.model_path
    )
    config = AutoConfig.from_pretrained(config_path)
    _propagate_config_fields(config, cfg)

    if cfg.inference.use_quantized_model:
        logger.info("Loading quantized model weights...")
        with torch.device("meta"):
            model = create_base_model(
                config,
                cfg.env.device,
                attention_backend=cfg.inference.attention_backend,
            )
    else:
        model = create_base_model(
            config,
            cfg.env.device,
            attention_backend=cfg.inference.attention_backend,
        )

    model = _load_weights(model, cfg, data_path)

    if backend == "megakernel_cuda":
        logger.info("Switching to CUDA megakernel backend...")
        model = create_megakernel_model(model, variant=cfg.inference.megakernel_variant)

    try:
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(data_path)
    except (OSError, ValueError):
        pass

    return model, tokenizer
