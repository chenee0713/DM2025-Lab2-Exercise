"""
DM2025 Lab 2 - Phase 3: XLM-RoBERTa Training
第三個模型：多語言 RoBERTa（架構差異最大化）
"""

import json
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
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
    MODEL_NAME = 'xlm-roberta-base'
    MAX_LENGTH = 128
    N_FOLDS = 5
    
    EPOCHS = 6
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    EARLY_STOPPING_PATIENCE = 3
    
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
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_competition_data():
    print("\n=== Data Loading ===")
    
    with open('data/final_posts.json', 'r', encoding='utf-8') as f:
        posts = json.load(f)
    
    data = []
    for item in posts:
        post_id = item['root']['_source']['post']['post_id']
        text = item['root']['_source']['post']['text']
        hashtags = item['root']['_source']['post']['hashtags']
        
        if hashtags:
            text = text + " " + " ".join([f"#{tag}" for tag in hashtags])
        
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
    print("\n=== XLM-RoBERTa Training ===")
    
    label2id = {label: idx for idx, label in enumerate(config.EMOTIONS)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    train_df['label'] = train_df['emotion'].map(label2id)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(config.EMOTIONS)),
        y=train_df['label'].values
    )
    class_weights = np.clip(class_weights, config.WEIGHT_MIN, config.WEIGHT_MAX)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
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
            label2id=label2id
        )
        
        train_dataset = EmotionDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
        val_dataset = EmotionDataset(X_val, y_val, tokenizer, config.MAX_LENGTH)
        
        training_args = TrainingArguments(
            output_dir=f"./results_xlm/fold_{fold}",
            num_train_epochs=config.EPOCHS,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            warmup_ratio=config.WARMUP_RATIO,
            logging_dir=f'./logs_xlm/fold_{fold}',
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
        
        os.makedirs('./xlm_models', exist_ok=True)
        model.save_pretrained(f'./xlm_models/fold_{fold}')
        tokenizer.save_pretrained(f'./xlm_models/fold_{fold}')
        
        del model, trainer, train_dataset, val_dataset
        torch.cuda.empty_cache()
    
    overall_f1 = f1_score(train_df['label'], oof_preds, average='macro')
    mean_f1 = np.mean(cv_scores)
    std_f1 = np.std(cv_scores)
    
    print(f"\n{'='*50}")
    print(f"XLM-RoBERTa CV Results:")
    for i, score in enumerate(cv_scores):
        print(f"  Fold {i+1}: {score:.4f}")
    print(f"-" * 50)
    print(f"  Mean F1:  {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  OOF F1:   {overall_f1:.4f}")
    print(f"{'='*50}")
    
    return overall_f1, oof_preds

def main():
    print("="*70)
    print("XLM-RoBERTa Training (Model #3)")
    print("="*70)
    
    train_df, test_df = load_competition_data()
    oof_f1, oof_preds = train_with_cross_validation(train_df, Config)
    
    print(f"\n{'='*70}")
    print(f"XLM-RoBERTa Training Complete")
    print(f"OOF F1: {oof_f1:.4f}")
    print(f"Models: ./xlm_models/fold_0-4/")
    print(f"\nNext: Run ensemble_3models.py")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()