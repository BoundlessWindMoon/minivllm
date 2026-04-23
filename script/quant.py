import os
import sys
import torch
import torch.distributed as dist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.logger import logger
from utils.config import GlobalConfig
from utils.model_loader import ModelLoader
from utils.quantizer import Quantizer

from transformers import AutoTokenizer, AutoConfig
from engine.model_runner import ModelRunner
from model.qwen3 import Qwen3ForCausalLM


def main():
    cfg = GlobalConfig.from_yaml("configs/default.yaml")

    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)

    dist.init_process_group(
        backend=cfg.env.distributed.backend if torch.cuda.is_available() else "gloo",
        init_method=cfg.env.distributed.init_method,
        world_size=cfg.env.distributed.world_size,
        rank=cfg.env.distributed.rank,
    )

    logger.info("Loading model...")
    data_path = cfg.path.data_path or cfg.path.model_path
    loader = ModelLoader(data_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.path.model_path)
    config = AutoConfig.from_pretrained(cfg.path.model_path)

    model_skeleton = Qwen3ForCausalLM(config).to(cfg.env.device)
    model = loader.inject_data(model_skeleton)

    quantizer = Quantizer(
        model=model,
        tokenizer=tokenizer,
        quant_config=cfg.quant,
        env_config=cfg.env,
    )

    logger.info("Starting quantization...")
    model = quantizer.run()

    runner = ModelRunner(model=model, tokenizer=tokenizer, cfg=cfg)

    text = runner.inference()
    logger.info(f"量化后生成结果: \n{text}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
