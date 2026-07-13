import argparse

import torch
import torch.distributed as dist

from utils.logger import logger
from utils.config import GlobalConfig, print_runtime_config, dump_config
from engine.model_runner import ModelRunner
from engine.loader import load_model
from engine.runtime_setup import apply_runtime_patches
from engine.processor import load_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runs/default.yaml")
    parser.add_argument("--model", default=None,
                        help="Override the model layer, e.g. 'qwen3_5' or a yaml path")
    parser.add_argument("--dump-config", action="store_true",
                        help="Print the fully-merged config and exit")
    args = parser.parse_args()
    cfg = GlobalConfig.from_yaml(args.config, model=args.model)

    if args.dump_config:
        print(dump_config(cfg))
        return

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
    model = apply_runtime_patches(model, cfg)
    processor = load_processor(cfg)
    runner = ModelRunner(model=model, tokenizer=tokenizer, processor=processor, cfg=cfg)

    text = runner.inference()
    logger.info(f"生成结果: \n{text}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
