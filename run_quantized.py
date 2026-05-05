import os
import json
import argparse
import torch
import torch.distributed as dist

from utils.logger import logger
from transformers import AutoTokenizer, AutoConfig
from utils.quant_model_io import (
    is_quantized_model,
    prepare_model_for_quantized_load,
    load_quantized_weights,
)
from engine.model_runner import ModelRunner
from model.qwen3 import Qwen3ForCausalLM
from utils.config import GlobalConfig


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

    quant_path = cfg.path.quantized_model_path or cfg.path.data_path
    if not quant_path:
        raise ValueError(
            "quantized_model_path or data_path must be set for quantized inference"
        )

    if not is_quantized_model(quant_path):
        raise ValueError(f"No quantized model found at {quant_path}")

    logger.info(f"Loading quantized model from {quant_path}...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)
    config = AutoConfig.from_pretrained(quant_path)
    config.use_sdpa = cfg.inference.use_sdpa
    model_skeleton = Qwen3ForCausalLM(config).to(cfg.env.device)

    with open(os.path.join(quant_path, "quant_config.json")) as f:
        quant_info = json.load(f)

    prepare_model_for_quantized_load(model_skeleton, quant_info, cfg.quant.backend)
    load_quantized_weights(
        model_skeleton, quant_path, quant_info, expected_config=cfg.quant
    )
    model = model_skeleton
    runner = ModelRunner(model=model, tokenizer=tokenizer, cfg=cfg)

    text = runner.inference()
    logger.info(f"量化模型生成结果: \n{text}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
