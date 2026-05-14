"""Entry point: unified inference for fp16 / bf16 / quantized / megakernel."""

import argparse

import torch
import torch.distributed as dist

from utils.logger import logger
from utils.config import GlobalConfig, print_runtime_config
from engine.model_runner import ModelRunner
from engine.loader import load_model


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
