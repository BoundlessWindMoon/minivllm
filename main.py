import os
import torch
import torch.distributed as dist
# import swanlab
from glob import glob
from torch import nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.ModelLoader import ModelLoader
from engine.ModelRunner import ModelRunner
from model.qwen3 import Qwen3ForCausalLM

device = "cuda:0"
max_new_tokens = 128

def main():
    
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://localhost:29500",
        world_size=1,
        rank=0
    )
    
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    data_path = model_path 
    
    msg = "Hello, my name is sakuya, im 24 year old"
    
    loader = ModelLoader(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    # model_skeleton = AutoModelForCausalLM.from_config(config).to(device)
    
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    
    model_skeleton = Qwen3ForCausalLM(config).to(device)
    
    model = loader.inject_data(model_skeleton)
    print(model)
    
    inputs = tokenizer(msg, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    runner = ModelRunner(model, device)

    # swanlab.init(
    #     project="nanovllm-sakuya",
        
    #     config={
    #         "architecture": "Qwen3-0.6B"
    #     }    
    # )
    
    # prefill
    logits = runner.compute_logits(input_ids, position_ids, is_prefill=True)
    
    # decode
    current_tokens = 0
    while(current_tokens < max_new_tokens):
        logits = runner.compute_logits(input_ids, position_ids, is_prefill=False)
        input_ids, position_ids = runner.post_process(input_ids, position_ids, logits)
        swanlab.log({"currents_tokens": current_tokens})
        current_tokens += 1

    current_text = tokenizer.decode(input_ids[0])
    print(f"当前生成: {current_text}")
    # swanlab.finish()

if __name__ == "__main__":
    main()