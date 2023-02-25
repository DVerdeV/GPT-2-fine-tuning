from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import torch

class MyModel(AutoModelForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.compute_loss(outputs.logits, labels)
        return {"loss": loss, "logits": logits}


tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-2-spanish")

with open("tema.txt", "r") as f:
    text = f.read()

dataset = load_dataset("text", data_files={"train": "tema.txt", "test": "tema.txt"})
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(text):
    inputs = tokenizer(text["text"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = inputs["input_ids"]
    inputs["attention_mask"] = [int(token_id != tokenizer.pad_token_id) for token_id in inputs["input_ids"]]
    return inputs

def data_collator(batch):
    input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
    labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


dataset = dataset.map(tokenize_function, batched=True)

model = MyModel.from_pretrained("flax-community/gpt-2-spanish")

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=20,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)
trainer.train()
trainer.save_model("./results")
# Ask a question to the model.
prompt = "La guerra de la independencia tiene su origen en"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))
