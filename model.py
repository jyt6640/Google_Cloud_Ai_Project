import zipfile
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import torch

# 데이터셋 경로
dataset_path = r'D:\Google_Cloud_Ai_Project\015.동화_줄거리_생성_데이터'
train_path = os.path.join(dataset_path, 'Training')
valid_path = os.path.join(dataset_path, 'Validation')

# 모든 zip 파일을 폴더 내에서 추출하는 함수
def extract_all_zips(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                extract_folder = zip_file_path.replace('.zip', '')
                os.makedirs(extract_folder, exist_ok=True)
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)

# JSON 데이터에서 문단 정보를 추출하는 함수
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

# 모든 JSON 데이터를 로드하는 함수
def load_all_json_data(folder_path):
    all_data = []
    seen = set()  # 중복된 문단을 추적하기 위한 집합
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    json_data = json.load(json_file)
                    data = extract_paragraph_info(json_data)
                    for entry in data:
                        text = entry['text']
                        if text not in seen:  # 중복 체크
                            seen.add(text)
                            all_data.append(entry)
    return all_data

# zip 파일 추출
extract_all_zips(train_path)
extract_all_zips(valid_path)

# JSON 데이터 로드
train_data = load_all_json_data(train_path)
valid_data = load_all_json_data(valid_path)

# 데이터프레임 생성 및 정렬
train_df = pd.DataFrame(train_data).sort_values(by=['title', 'page'])
valid_df = pd.DataFrame(valid_data).sort_values(by=['title', 'page'])

# Hugging Face 데이터셋 생성
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)

# 토크나이즈 함수 정의
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 데이터셋 토크나이즈 및 형식 지정
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 로깅 콜백 클래스 정의
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        print(logs)

# 트레이닝 아규먼트 설정
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,  # 에폭 수를 1로 설정
    per_device_train_batch_size=1,  # 배치 크기를 1로 설정
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',  # 로그 저장 디렉토리
    logging_steps=100,  # 100 스텝마다 로그 기록
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[LoggingCallback]  # 커스텀 콜백 추가
)

# 모델 트레이닝
trainer.train()

# 모델 저장
try:
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
except Exception as e:
    print(f"모델 저장 중 오류 발생: {e}")
