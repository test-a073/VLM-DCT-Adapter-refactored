# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",torch_dtype=torch.float16)
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

print(f"[Model]\n",model)
print(f"[PAD token]", tokenizer.pad_token)

if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token # Set pad token if not pres
	print(f"[PAD token (none->to something)]", tokenizer.pad_token)