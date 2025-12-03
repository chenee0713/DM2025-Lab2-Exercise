"""
DM2025 Lab 2 - Phase 3: 3-Model Ensemble
結合 Twitter-RoBERTa + DeBERTa + XLM-RoBERTa
"""

import json
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
N_FOLDS = 5
BATCH_SIZE = 16

class EmotionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def clean_text_roberta(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def clean_text_basic(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_test_data():
    print("\n=== Loading Test Data ===")
    
    with open('data/final_posts.json', 'r', encoding='utf-8') as f:
        posts = json.load(f)
    
    # Twitter-RoBERTa 版本
    data_roberta = []
    for item in posts:
        post_id = item['root']['_source']['post']['post_id']
        text = item['root']['_source']['post']['text']
        hashtags = item['root']['_source']['post']['hashtags']
        
        if hashtags:
            text = text + " " + " ".join([f"#{tag}" for tag in hashtags])
        
        text = clean_text_roberta(text)
        data_roberta.append({'id': post_id, 'text': text})
    
    # DeBERTa/XLM 版本
    data_other = []
    for item in posts:
        post_id = item['root']['_source']['post']['post_id']
        text = item['root']['_source']['post']['text']
        hashtags = item['root']['_source']['post']['hashtags']
        
        if hashtags:
            text = text + " " + " ".join([f"#{tag}" for tag in hashtags])
        
        text = clean_text_basic(text)
        data_other.append({'id': post_id, 'text': text})
    
    df_roberta = pd.DataFrame(data_roberta)
    df_other = pd.DataFrame(data_other)
    
    split_df = pd.read_csv('data/data_identification.csv')
    
    df_roberta = df_roberta.merge(split_df, on='id', how='left')
    df_other = df_other.merge(split_df, on='id', how='left')
    
    test_roberta = df_roberta[df_roberta['split'] == 'test'].reset_index(drop=True)
    test_other = df_other[df_other['split'] == 'test'].reset_index(drop=True)
    
    print(f"Test samples: {len(test_roberta)}")
    
    return test_roberta, test_other

def predict_model(test_df, model_dir, model_name):
    print(f"\n=== Predicting {model_name} ===")
    
    # 定義每個模型對應的原始 Tokenizer 名稱
    tokenizer_map = {
        'Twitter-RoBERTa': 'cardiffnlp/twitter-roberta-base-emotion',
        'DeBERTa': 'microsoft/deberta-v3-base',
        'XLM-RoBERTa': 'xlm-roberta-base'
    }
    
    base_model_id = tokenizer_map.get(model_name)
    
    if base_model_id is None:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"Loading Tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    all_probs = []
    
    for fold in range(N_FOLDS):
        print(f"Loading Fold {fold + 1}...")
        
        # 使用絕對路徑
        model_path = os.path.abspath(f'{model_dir}/fold_{fold}')
        print(f"Model path: {model_path}")
        
        # 移除 local_files_only 參數
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        test_dataset = EmotionDataset(test_df['text'].values, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold+1}", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                fold_probs.append(probs.cpu().numpy())
        
        fold_probs = np.vstack(fold_probs)
        all_probs.append(fold_probs)
        
        del model
        torch.cuda.empty_cache()
    
    model_probs = np.mean(all_probs, axis=0)
    print(f"{model_name} complete")
    
    return model_probs

def ensemble_3models(roberta_probs, deberta_probs, xlm_probs, 
                     w1=0.33, w2=0.33, w3=0.34):
    """
    3-Model Ensemble
    
    Args:
        w1: Twitter-RoBERTa weight
        w2: DeBERTa weight
        w3: XLM-RoBERTa weight
    """
    print(f"\n=== 3-Model Ensemble ===")
    print(f"  Twitter-RoBERTa: {w1:.2f}")
    print(f"  DeBERTa:         {w2:.2f}")
    print(f"  XLM-RoBERTa:     {w3:.2f}")
    
    ensemble_probs = (
        roberta_probs * w1 + 
        deberta_probs * w2 + 
        xlm_probs * w3
    )
    
    final_preds = np.argmax(ensemble_probs, axis=-1)
    predictions = [EMOTIONS[pred] for pred in final_preds]
    
    return predictions, ensemble_probs

def create_submission(test_df, predictions, filename='submission_3models.csv'):
    print(f"\n=== Creating Submission ===")
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'emotion': predictions
    })
    
    submission.to_csv(filename, index=False)
    
    print(f"File: {filename}")
    print(f"Total: {len(submission)} rows")
    
    print(f"\nPrediction Distribution:")
    dist = submission['emotion'].value_counts()
    for emotion, count in dist.items():
        pct = count / len(submission) * 100
        print(f"  {emotion:10s}: {count:5d} ({pct:5.2f}%)")
    
    return submission

def main():
    print("="*70)
    print("3-Model Ensemble: RoBERTa + DeBERTa + XLM-RoBERTa")
    print("="*70)
    
    test_roberta, test_other = load_test_data()
    
    # 預測三個模型
    roberta_probs = predict_model(test_roberta, './best_models', 'Twitter-RoBERTa')
    deberta_probs = predict_model(test_other, './deberta_models', 'DeBERTa')
    xlm_probs = predict_model(test_other, './xlm_models', 'XLM-RoBERTa')
    
    # 3-Model Ensemble（平均）
    predictions, ensemble_probs = ensemble_3models(
        roberta_probs, 
        deberta_probs,
        xlm_probs,
        w1=0.33, w2=0.33, w3=0.34  # 可調整
    )
    
    submission = create_submission(test_roberta, predictions)
    
    print(f"\n{'='*70}")
    print(f"3-Model Ensemble Complete")
    print(f"{'='*70}")
    print(f"\nOutput: submission_3models.csv")
    print(f"\nExpected: +1-2% over 2-model ensemble")
    print(f"Baseline: 0.6981 → Target: 0.710+")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()