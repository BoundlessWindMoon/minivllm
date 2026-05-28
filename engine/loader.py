"""Top-level model loader.

Handles: tokenizer load, quantized vs fp/bf16 branch, meta-skeleton
materialization with optional CPU offload, megakernel backend switch,
generation_config attachment.
"""

import os
import json

import torch
from transformers import AutoTokenizer, AutoConfig, AutoProcessor

from utils.logger import logger
from utils.config import GlobalConfig, resolve_data_path
from utils.model_loader import ModelLoader
from utils.cpu_offload import materialize_with_offload, apply_cpu_offload
from quantization.checkpoint import (
    prepare_model_for_quantized_load,
    load_quantized_weights,
)
from model.factory import create_base_model, create_megakernel_model


def load_model(cfg: GlobalConfig):
    data_path = resolve_data_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)
    backend = cfg.inference.backend

    # Propagate linear-attn backend preference into the model config so that
    # layers can read it at construction time.
    _la_backend = getattr(cfg.inference, "linear_attn_prefill_backend", "torch")
    _la_decode_backend = getattr(cfg.inference, "linear_attn_decode_backend", "fla")

    if cfg.inference.use_quantized_model:
        if backend == "megakernel_cuda":
            raise RuntimeError(
                "Megakernel backend does not support quantized models. "
                "Please set inference.backend to 'default' when using a quantized model."
            )
        logger.info("Loading quantized model weights...")
        config = AutoConfig.from_pretrained(data_path)
        config.linear_attn_prefill_backend = _la_backend
        config.linear_attn_decode_backend = _la_decode_backend
        _kv_max = getattr(cfg.inference, "kv_cache_max_len", None)
        if _kv_max is not None:
            config.kv_cache_max_len = _kv_max
        with torch.device("meta"):
            model = create_base_model(config, cfg.env.device, attention_backend=cfg.inference.attention_backend)
        with open(os.path.join(data_path, "quant_config.json")) as f:
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
        load_quantized_weights(
            model, data_path, quant_info, expected_config=cfg.quant
        )
    else:
        loader = ModelLoader(data_path)
        config = AutoConfig.from_pretrained(cfg.path.model_path)
        config.linear_attn_prefill_backend = _la_backend
        config.linear_attn_decode_backend = _la_decode_backend
        _kv_max = getattr(cfg.inference, "kv_cache_max_len", None)
        if _kv_max is not None:
            config.kv_cache_max_len = _kv_max
        model = create_base_model(config, cfg.env.device, attention_backend=cfg.inference.attention_backend)
        model = loader.inject_data(model)

    if backend == "megakernel_cuda":
        logger.info("Switching to CUDA megakernel backend...")
        model = create_megakernel_model(
            model, variant=cfg.inference.megakernel_variant
        )

        sampling = cfg.inference.sampling
        if (
            sampling.sample_method == "greedy"
            and sampling.temperature == 1.0
            and sampling.topp == 1.0
        ):
            model.greedy_fast_path = True
            logger.info("Megakernel: enabled greedy fast path (kernel argmax only)")

    try:
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(data_path)
    except (OSError, ValueError):
        pass

    # Inject KV-cache backend if requested.
    if getattr(cfg.inference, "kv_cache", None) and cfg.inference.kv_cache.backend == "kivi":
        from layers.kv_cache import KiviKVCacheBackend

        kv_cfg = cfg.inference.kv_cache
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None:
            for layer in layers:
                attn_module = getattr(getattr(layer, "self_attn", None), "attn", None)
                if attn_module is not None and hasattr(attn_module, "kv_backend"):
                    attn_module.kv_backend = KiviKVCacheBackend(
                        batch_size=1,
                        num_kv_heads=attn_module.num_kv_heads,
                        max_seq_len=attn_module.max_seq_len,
                        head_dim=attn_module.head_dim,
                        k_bits=kv_cfg.k_bits,
                        v_bits=kv_cfg.v_bits,
                        group_size=kv_cfg.group_size,
                        residual_length=kv_cfg.residual_length,
                        device=cfg.env.device,
                        dtype=cfg.env.get_torch_dtype(),
                    )
            logger.info(
                f"[KVCache] Injected KiviKVCacheBackend "
                f"(k_bits={kv_cfg.k_bits}, v_bits={kv_cfg.v_bits}, "
                f"group_size={kv_cfg.group_size}, residual={kv_cfg.residual_length})"
            )

    processor = None
    if getattr(cfg.inference, "multimodal", None) and cfg.inference.multimodal.enabled:
        processor = AutoProcessor.from_pretrained(cfg.path.model_path, trust_remote_code=True)
        logger.info("Loaded AutoProcessor for multimodal inference.")

    return model, tokenizer, processor
