import zipfile
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import torch

# Paths to the dataset
dataset_path = r'C:\Users\admin\Desktop\Google_Cloud_Ai_Project\015.동화_줄거리_생성_데이터'
train_path = os.path.join(dataset_path, 'Training')
valid_path = os.path.join(dataset_path, 'Validation')

def extract_all_zips(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                extract_folder = zip_file_path.replace('.zip', '')
                os.makedirs(extract_folder, exist_ok=True)
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)

def extract_paragraph_info(json_data):
    title = json_data.get('title', 'Unknown Title')
    paragraphs = json_data.get('paragraphInfo', [])
    data = []
    for para in paragraphs:
        data.append({
            'title': title,
            'text': para['srcText'],
            'page': para['srcPage'],
            'sentences': para['srcSentenceEA'],
            'words': para['srcWordEA']
        })
    return data

def load_all_json_data(folder_path):
    all_data = []
    seen = set()  # To track unique paragraphs
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    json_data = json.load(json_file)
                    data = extract_paragraph_info(json_data)
                    for entry in data:
                        text = entry['text']
                        if text not in seen:  # Check for duplicates
                            seen.add(text)
                            all_data.append(entry)
    return all_data

extract_all_zips(train_path)
extract_all_zips(valid_path)

train_data = load_all_json_data(train_path)
valid_data = load_all_json_data(valid_path)

train_df = pd.DataFrame(train_data).sort_values(by=['title', 'page'])
valid_df = pd.DataFrame(valid_data).sort_values(by=['title', 'page'])

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

tokenizer = AutoTokenizer.from_pretrained("kihoonlee/STOCK_SOLAR-10.7B")
model = AutoModelForCausalLM.from_pretrained("kihoonlee/STOCK_SOLAR-10.7B")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        print(logs)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=100,  # Log every 100 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[LoggingCallback]  # Add the custom callback here
)

trainer.train()

try:
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
except Exception as e:
    print(f"Error saving the model: {e}")
