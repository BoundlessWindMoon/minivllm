from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    msg = "Hello, my name is "
    tokenizer = AutoTokenizer.from_pretrained("~/huggingface/Qwen3-0.6B/")
    model = AutoModelForCausalLM.from_pretrained("~/huggingface/Qwen3-0.6B/", 
                                                 device_map="auto")
    inputs = tokenizer(msg, return_tensors="pt")
    output = model.generate(input_ids=inputs["input_ids"], max_length=1000)
    print(output)

if __name__ == "__main__":
    main()