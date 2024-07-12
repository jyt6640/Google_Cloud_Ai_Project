import zipfile
import os
import json
import pandas as pd

dataset_path = r'C:\Users\user\Desktop\Google_Cloud_Ai_Project\015.동화_줄거리_생성_데이터'
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
    seen = set() 
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    json_data = json.load(json_file)
                    data = extract_paragraph_info(json_data)
                    for entry in data:
                        text = entry['text']
                        if text not in seen: 
                            seen.add(text)
                            all_data.append(entry)
    return all_data

extract_all_zips(train_path)
extract_all_zips(valid_path)

train_data = load_all_json_data(train_path)
valid_data = load_all_json_data(valid_path)

train_df = pd.DataFrame(train_data).sort_values(by=['title', 'page'])
valid_df = pd.DataFrame(valid_data).sort_values(by=['title', 'page'])

train_df.to_csv('training_data.csv', index=False)
valid_df.to_csv('validation_data.csv', index=False)

train_df.head()
