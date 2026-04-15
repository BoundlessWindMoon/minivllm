import os
import torch
import torch.distributed as dist

from utils.logger import logger
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.model_loader import ModelLoader
from engine.model_runner import ModelRunner
from model.qwen3 import Qwen3ForCausalLM

device = "cuda:0"
max_new_tokens = 128
msg = "Hello my name is sakuya, im 24 year old and study in UCAS University"
model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
baseline_model_path = os.path.expanduser("~/huggingface/baseline/")
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
data_path = model_path


def main():

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://localhost:29500",
        world_size=1,
        rank=0,
    )

    logger.info("Loading model...")
    loader = ModelLoader(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    model_skeleton = Qwen3ForCausalLM(config).to(device)
    model = loader.inject_data(model_skeleton)
    runner = ModelRunner(
        model=model,
        tokenizer=tokenizer,
        prompt=msg,
        device=device,
        max_new_tokens=max_new_tokens,
        check_correction=True,
        use_profile=False,
        baseline_model_path=baseline_model_path,
    )

    text = runner.inference()
    logger.info(f"生成结果: \n{text}")


if __name__ == "__main__":
    main()
