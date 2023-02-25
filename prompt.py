from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Import model from folder "results"
model = AutoModelForCausalLM.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-2-spanish")

# Ask a question to the model.
prompt = "La primera programadora fue"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
