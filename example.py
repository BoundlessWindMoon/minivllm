from transformers import AutoModelForCausalLM, AutoTokenizer
import os

device = "cuda:0"

def main():
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    msg = "Hello, my name is "
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 device_map=device)
    inputs = tokenizer(msg, return_tensors="pt").to(device)
    output = model.generate(input_ids=inputs["input_ids"], max_length=1000,temperature=2.0, do_sample=True)
    print(tokenizer.decode(inputs["input_ids"], skip_special_tokens=True))
    print(tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))

if __name__ == "__main__":
    main()