import os
import torch
from glob import glob
from torch import nn
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = "cuda:0"
max_new_tokens = 128

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

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
    
    msg = "Hello, my name is "
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config).to(device)
    load_model(model, model_path)
    
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