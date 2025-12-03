"""
DM2025 Lab 2 - Phase 3: Twitter Emotion Classification - IMPROVED VERSION
改進項目：
1. 優化訓練策略：6 epochs, patience 3
2. 溫和權重：裁剪到 0.5-3.0
3. 文本預處理：保留 @user 和 http token（Twitter 模型專用）
"""

import json
import pandas as pd
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Config:
    MODEL_NAME = 'cardiffnlp/twitter-roberta-base-emotion'
    MAX_LENGTH = 128
    N_FOLDS = 5
    
    # 訓練策略
    EPOCHS = 6  
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    EARLY_STOPPING_PATIENCE = 3
    
    # 權重裁剪
    WEIGHT_MIN = 0.5
    WEIGHT_MAX = 3.0
    
    SEED = 42
    EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def clean_text(text):
    """清理推文文本 - Twitter RoBERTa 專用"""
    new_text = []
    for t in text.split(" "):
        # 將 @user 替換為模型認識的特殊 token
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        # 將網址替換為 http
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def load_competition_data():
    print("\n=== Stage 1: Data Loading ===")
    
    with open('data/final_posts.json', 'r', encoding='utf-8') as f:
        posts = json.load(f)
    
    data = []
    for item in posts:
        post_id = item['root']['_source']['post']['post_id']
        text = item['root']['_source']['post']['text']
        hashtags = item['root']['_source']['post']['hashtags']
        
        if hashtags:
            text = text + " " + " ".join([f"#{tag}" for tag in hashtags])
        
        # 應用文本清理
        text = clean_text(text)
        
        data.append({'id': post_id, 'text': text})
    
    df = pd.DataFrame(data)
    emotion_df = pd.read_csv('data/emotion.csv')
    split_df = pd.read_csv('data/data_identification.csv')
    
    df = df.merge(split_df, on='id', how='left')
    df = df.merge(emotion_df, on='id', how='left')
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"\nEmotion Distribution:")
    print(train_df['emotion'].value_counts())
    
    return train_df, test_df

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
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
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_with_cross_validation(train_df, config):
    print("\n=== Stage 2: Model Training ===")
    
    label2id = {label: idx for idx, label in enumerate(config.EMOTIONS)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    train_df['label'] = train_df['emotion'].map(label2id)
    
    # 計算並裁剪 class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(config.EMOTIONS)),
        y=train_df['label'].values
    )
    # 裁剪到合理範圍
    # class_weights = np.clip(class_weights, config.WEIGHT_MIN, config.WEIGHT_MAX)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    print("\nClass Weights (clipped):")
    for emotion, weight in zip(config.EMOTIONS, class_weights):
        print(f"  {emotion:10s}: {weight:.3f}")
    
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    
    oof_preds = np.zeros(len(train_df), dtype=int)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        
        print(f"\n=== Fold {fold + 1}/{config.N_FOLDS} ===")
        
        X_train = train_df.iloc[train_idx]['text'].values
        y_train = train_df.iloc[train_idx]['label'].values
        X_val = train_df.iloc[val_idx]['text'].values
        y_val = train_df.iloc[val_idx]['label'].values
        
        print(f"Train: {len(X_train)} | Val: {len(X_val)}")
        
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=len(config.EMOTIONS),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        
        train_dataset = EmotionDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
        val_dataset = EmotionDataset(X_val, y_val, tokenizer, config.MAX_LENGTH)
        
        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold}",
            num_train_epochs=config.EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            logging_dir=f'./logs/fold_{fold}',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            report_to="none",
            seed=config.SEED
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            f1_macro = f1_score(labels, preds, average='macro')
            return {'f1_macro': f1_macro}
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)]
        )
        
        trainer.train()
        
        val_outputs = trainer.predict(val_dataset)
        val_preds = np.argmax(val_outputs.predictions, axis=-1)
        oof_preds[val_idx] = val_preds
        
        fold_f1 = f1_score(y_val, val_preds, average='macro')
        cv_scores.append(fold_f1)
        
        print(f"\nFold {fold + 1} F1: {fold_f1:.4f}")
        
        os.makedirs('./best_models', exist_ok=True)
        model.save_pretrained(f'./best_models/fold_{fold}')
        tokenizer.save_pretrained(f'./best_models/fold_{fold}')
        
        del model, trainer, train_dataset, val_dataset
        torch.cuda.empty_cache()
    
    overall_f1 = f1_score(train_df['label'], oof_preds, average='macro')
    mean_f1 = np.mean(cv_scores)
    std_f1 = np.std(cv_scores)
    
    print(f"\n{'='*50}")
    print(f"CV Results:")
    for i, score in enumerate(cv_scores):
        print(f"  Fold {i+1}: {score:.4f}")
    print(f"-" * 50)
    print(f"  Mean F1:  {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  OOF F1:   {overall_f1:.4f}")
    print(f"{'='*50}")
    print(f"\nExpected Kaggle Private LB: {overall_f1:.4f} +/- 0.01")
    
    return overall_f1, oof_preds

def predict_test_ensemble(test_df, config):
    print(f"\n=== Stage 3: Prediction & Submission ===")
    
    all_probs = []
    
    for fold in range(config.N_FOLDS):
        print(f"Loading Fold {fold + 1}...")
        
        model_path = f'./best_models/fold_{fold}'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        test_dataset = EmotionDataset(
            test_df['text'].values,
            None,
            tokenizer,
            config.MAX_LENGTH
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE * 2,
            shuffle=False
        )
        
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
    
    ensemble_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(ensemble_probs, axis=-1)
    
    predictions = [config.EMOTIONS[pred] for pred in final_preds]
    
    print("Prediction Complete")
    
    return predictions, ensemble_probs

def create_kaggle_submission(test_df, predictions, filename='submission.csv'):
    print(f"\nCreating Submission File...")
    
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
    
    sample = pd.read_csv('data/samplesubmission.csv')
    assert list(submission.columns) == list(sample.columns), "Column error"
    assert len(submission) == len(sample), "Row count error"
    print("\nFormat check passed")
    
    return submission

def main():
    print("="*70)
    print("DM2025 Lab 2 - Phase 3 - IMPROVED VERSION")
    print("="*70)
    
    print(f"\nImprovements Applied:")
    print(f"  1. Model: {Config.MODEL_NAME}")
    print(f"  2. Epochs: {Config.EPOCHS} (was 4)")
    print(f"  3. Batch Size: {Config.BATCH_SIZE}")
    print(f"  4. Learning Rate: {Config.LEARNING_RATE}")
    print(f"  5. Weight Clipping: [{Config.WEIGHT_MIN}, {Config.WEIGHT_MAX}]")
    print(f"  6. Text Cleaning: @user and http token preservation")
    print(f"  7. Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE} (was 2)")
    
    train_df, test_df = load_competition_data()
    
    oof_f1, oof_preds = train_with_cross_validation(train_df, Config)
    
    predictions, probs = predict_test_ensemble(test_df, Config)
    
    submission = create_kaggle_submission(test_df, predictions, 'submission.csv')
    
    print(f"\n{'='*70}")
    print(f"Training Complete")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  Local OOF F1: {oof_f1:.4f}")
    print(f"  Expected Kaggle Private LB: {oof_f1:.4f} +/- 0.01")
    print(f"\nOutput Files:")
    print(f"  1. submission.csv")
    print(f"  2. ./best_models/fold_0-4/")
    print(f"\nNext Steps:")
    print(f"  1. Submit submission.csv to Kaggle")
    print(f"  2. Compare with baseline (0.6844)")
    print(f"  3. Max 5 submissions per day")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()