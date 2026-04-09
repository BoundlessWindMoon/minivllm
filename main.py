import os
import torch
from glob import glob
from torch import nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.ModelLoader import ModelLoader

device = "cuda:0"
max_new_tokens = 128

def compute_logits(model, input_ids: torch.Tensor, position: torch.Tensor, is_prefill: bool) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids.to(device),
        position_ids=position.to(device),
        use_cache=False,
        past_key_values=None
    )
    return outputs.logits

def post_process(input_ids: torch.Tensor, position: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    new_input_ids = torch.cat([input_ids, next_token], dim=-1)
    new_position = position[:, -1:] + 1
    new_position_ids = torch.cat([position, new_position], dim=-1)
    return new_input_ids, new_position_ids

def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    data_path = model_path 
    
    msg = "Hello, my name is "
    
    loader = ModelLoader(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model_skeleton = AutoModelForCausalLM.from_config(config).to(device)
    model = loader.inject_data(model_skeleton)
    
    print(model)
    
    inputs = tokenizer(msg, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    print(inputs["input_ids"])

    # prefill
    logits = compute_logits(model, input_ids, position_ids, is_prefill=True)
    
    # decode
    current_tokens = 0
    
    while(current_tokens < max_new_tokens):
        logits = compute_logits(model, input_ids, position_ids, is_prefill=False)
        input_ids, position_ids = post_process(input_ids, position_ids, logits)
        current_tokens += 1

    current_text = tokenizer.decode(input_ids[0])
    print(f"当前生成: {current_text}")
    

if __name__ == "__main__":
    main()