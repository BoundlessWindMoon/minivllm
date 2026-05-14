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
from model.qwen3 import Qwen3ForCausalLM


def load_model(cfg: GlobalConfig):
    data_path = resolve_data_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)
    backend = cfg.inference.backend

    if cfg.inference.use_quantized_model:
        if backend == "megakernel_cuda":
            raise RuntimeError(
                "Megakernel backend does not support quantized models. "
                "Please set inference.backend to 'default' when using a quantized model."
            )
        logger.info("Loading quantized model weights...")
        config = AutoConfig.from_pretrained(data_path)
        config.use_sdpa = cfg.inference.use_sdpa
        with torch.device("meta"):
            model_skeleton = Qwen3ForCausalLM(config)
        with open(os.path.join(data_path, "quant_config.json")) as f:
            quant_info = json.load(f)
        prepare_model_for_quantized_load(model_skeleton, quant_info, cfg.quant.backend)
        offload_paths = list(cfg.inference.cpu_offload_modules)
        materialize_with_offload(model_skeleton, cfg.env.device, offload_paths)
        if offload_paths:
            apply_cpu_offload(model_skeleton, offload_paths, cfg.env.device)
            logger.info(f"CPU offload enabled for: {offload_paths}")
            if cfg.inference.use_cuda_graph:
                logger.warning(
                    "CPU offload is incompatible with CUDA graph capture "
                    "(stream capture forbids host<->device copies and CPU "
                    "kernels); disabling use_cuda_graph for this run."
                )
                cfg.inference.use_cuda_graph = False
        load_quantized_weights(
            model_skeleton, data_path, quant_info, expected_config=cfg.quant
        )
        model = model_skeleton
    else:
        loader = ModelLoader(data_path)
        config = AutoConfig.from_pretrained(cfg.path.model_path)
        config.use_sdpa = cfg.inference.use_sdpa
        model_skeleton = Qwen3ForCausalLM(config).to(cfg.env.device)
        model = loader.inject_data(model_skeleton)

    if backend == "megakernel_cuda":
        logger.info("Switching to CUDA megakernel backend...")
        from model.qwen3_megakernel import Qwen3MegakernelForCausalLM

        model = Qwen3MegakernelForCausalLM.from_model(
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

    return model, tokenizer
