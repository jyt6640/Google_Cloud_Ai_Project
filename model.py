from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Save DataFrame to CSV
train_df.to_csv('training_data.csv', index=False)
valid_df.to_csv('validation_data.csv', index=False)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("kihoonlee/STOCK_SOLAR-10.7B")
model = AutoModelForCausalLM.from_pretrained("kihoonlee/STOCK_SOLAR-10.7B")

# Create dataset
def load_dataset(file_path, tokenizer):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )
    return dataset

# Load dataset
train_dataset = load_dataset('training_data.csv', tokenizer)
valid_dataset = load_dataset('validation_data.csv', tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
