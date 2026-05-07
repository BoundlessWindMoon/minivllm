import os
import json
import argparse
import torch
import torch.distributed as dist

from utils.logger import logger
from transformers import AutoTokenizer, AutoConfig
from utils.model_loader import ModelLoader
from utils.quant_model_io import (
    prepare_model_for_quantized_load,
    load_quantized_weights,
)
from engine.model_runner import ModelRunner
from model.qwen3 import Qwen3ForCausalLM
from utils.config import (
    GlobalConfig,
    resolve_data_path,
    print_runtime_config,
)


def load_model(cfg):
    data_path = resolve_data_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)
    backend = cfg.inference.backend

    if cfg.inference.use_quanted_model:
        if backend == "megakernel_cuda":
            raise RuntimeError(
                "Megakernel backend does not support quantized models. "
                "Please set inference.backend to 'default' when using a quantized model."
            )
        logger.info("Loading quantized model weights...")
        config = AutoConfig.from_pretrained(data_path)
        config.use_sdpa = cfg.inference.use_sdpa
        model_skeleton = Qwen3ForCausalLM(config).to(cfg.env.device)
        with open(os.path.join(data_path, "quant_config.json")) as f:
            quant_info = json.load(f)
        prepare_model_for_quantized_load(model_skeleton, quant_info, cfg.quant.backend)
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

        model = Qwen3MegakernelForCausalLM.from_model(model)

        sampling = cfg.inference.sampling
        if (
            sampling.sample_method == "greedy"
            and sampling.temperature == 1.0
            and sampling.topp == 1.0
        ):
            model.greedy_fast_path = True
            logger.info("Megakernel: enabled greedy fast path (kernel argmax only)")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = GlobalConfig.from_yaml(args.config)

    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)

    dist.init_process_group(
        backend=cfg.env.distributed.backend if torch.cuda.is_available() else "gloo",
        init_method=cfg.env.distributed.init_method,
        world_size=cfg.env.distributed.world_size,
        rank=cfg.env.distributed.rank,
    )

    print_runtime_config(cfg)

    model, tokenizer = load_model(cfg)
    runner = ModelRunner(model=model, tokenizer=tokenizer, cfg=cfg)

    text = runner.inference()
    logger.info(f"生成结果: \n{text}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
