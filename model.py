import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import torch
import time

# CSV 파일 경로
train_csv_file_path = r'D:\Google_Cloud_Ai_Project\training_data.csv'
valid_csv_file_path = r'D:\Google_Cloud_Ai_Project\validation_data.csv'

# CSV 파일 로드
train_df = pd.read_csv(train_csv_file_path)
valid_df = pd.read_csv(valid_csv_file_path)

# 데이터셋 크기 줄이기 (필요시)
train_df = train_df[:1000]
valid_df = valid_df[:1000]

# 데이터프레임 생성 및 정렬
train_df = train_df.sort_values(by=['title', 'page'])
valid_df = valid_df.sort_values(by=['title', 'page'])

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

# 첫 10 스텝 동안의 시간을 측정하여 전체 시간을 예측하는 콜백
class TimeLoggingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.total_steps = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 1:
            self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.total_steps += 1
        if self.total_steps == 10:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            est_total_time = elapsed_time * (args.max_steps / 10)
            print(f"Estimated total training time: {est_total_time / 3600:.2f} hours")

# 트레이닝 아규먼트 설정
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,  # 에폭 수를 3로 설정
    per_device_train_batch_size=8,  # 배치 크기를 8로 설정
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',  # 로그 저장 디렉토리
    logging_steps=100,  # 100 스텝마다 로그 기록
    fp16=True,  # 혼합 정밀도 학습을 사용하여 GPU 메모리 절약 및 학습 속도 향상
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[LoggingCallback(), TimeLoggingCallback()]  # 커스텀 콜백 추가
)

# 모델 트레이닝
trainer.train()

# 모델 저장
try:
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')
except Exception as e:
    print(f"모델 저장 중 오류 발생: {e}")
