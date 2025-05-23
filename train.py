from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")


def get_dataset(file_path, tokenizer):
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)

dataset = get_dataset("data.txt", tokenizer)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=2
)


trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=dataset
)
trainer.train()