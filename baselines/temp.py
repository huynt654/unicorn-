from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-Nemo-Base-2407"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Hello my name is", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# 'Qwen/Qwen2-72B'
