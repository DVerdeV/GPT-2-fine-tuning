from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-2-spanish")

with open("tema1.txt", "r") as f:
    text = f.read()

dataset = load_dataset("text", data_files={"train": text})
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(text):
    return tokenizer(text["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("flax-community/gpt-2-spanish")

training_args = TrainingArguments(output_dir="./results")

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

trainer.train()
