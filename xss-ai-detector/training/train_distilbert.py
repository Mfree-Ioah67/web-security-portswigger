"""
Train DistilBERT for XSS detection.

Features used:
    - DistilBERT tokenized input (max_len=128)
    - 52 handcrafted features appended to [CLS] embedding
      via a custom two-input classification head

Run:
    python training/train_distilbert.py
GPU recommended — CPU will be very slow.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocess import load_and_split
from utils.evaluate import get_metrics, print_report, plot_confusion_matrix
from utils.features import get_transformer_extra_features

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'xss_dataset_500.csv')
MODEL_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'models', 'distilbert')
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'models', 'results_distil')

EPOCHS     = 5
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 128
N_EXTRA    = 52   # handcrafted feature dimension


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class XSSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, extra_feats, max_len=MAX_LEN):
        self.texts       = list(texts)
        self.labels      = list(labels)
        self.tokenizer   = tokenizer
        self.extra_feats = extra_feats   # np.ndarray (n, 52)
        self.max_len     = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'extra':          torch.tensor(self.extra_feats[idx], dtype=torch.float),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model: DistilBERT [CLS] + handcrafted features -> classifier
# ---------------------------------------------------------------------------
class DistilBertWithFeatures(nn.Module):
    def __init__(self, n_extra=N_EXTRA, n_labels=2, dropout=0.3):
        super().__init__()
        self.bert      = DistilBertModel.from_pretrained('distilbert-base-uncased')
        hidden         = self.bert.config.hidden_size   # 768
        self.dropout   = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + n_extra, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_labels),
        )

    def forward(self, input_ids, attention_mask, extra):
        out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls   = out.last_hidden_state[:, 0, :]          # [CLS] token
        cls   = self.dropout(cls)
        fused = torch.cat([cls, extra], dim=1)          # concat handcrafted
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cpu':
        print('Warning: no GPU detected — training will be slow.')

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATASET_PATH)

    # Handcrafted features (normalized)
    print('\nExtracting handcrafted features...')
    extra_train = get_transformer_extra_features(X_train)
    extra_val   = get_transformer_extra_features(X_val)
    extra_test  = get_transformer_extra_features(X_test)
    print(f'Handcrafted feature shape: {extra_train.shape}')

    print('\nLoading DistilBERT tokenizer...')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_ds = XSSDataset(X_train, y_train, tokenizer, extra_train)
    val_ds   = XSSDataset(X_val,   y_val,   tokenizer, extra_val)
    test_ds  = XSSDataset(X_test,  y_test,  tokenizer, extra_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)
    test_loader  = DataLoader(test_ds,  batch_size=32)

    model     = DistilBertWithFeatures().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1, patience, patience_limit = 0.0, 0, 2

    print(f'\nTraining DistilBERT + Handcrafted ({EPOCHS} epochs)...')
    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ext  = batch['extra'].to(device)
            lbl  = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask, ext)
            loss   = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # --- validate ---
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                ext  = batch['extra'].to(device)
                out  = model(ids, mask, ext)
                val_preds.extend(out.argmax(-1).cpu().tolist())

        m = get_metrics(y_val, val_preds)
        print(f'Epoch {epoch}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}'
              f'  val_acc={m["accuracy"]:.4f}  val_f1={m["f1"]:.4f}')

        if m['f1'] > best_val_f1:
            best_val_f1 = m['f1']
            patience    = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
        else:
            patience += 1
            if patience >= patience_limit:
                print(f'Early stopping at epoch {epoch}.')
                break

    # --- Final test evaluation ---
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt'),
                                     map_location=device))
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            ext  = batch['extra'].to(device)
            out  = model(ids, mask, ext)
            test_preds.extend(out.argmax(-1).cpu().tolist())

    y_pred = np.array(test_preds)
    print(f'\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print_report(y_test, y_pred, 'DistilBERT + Handcrafted')

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion_matrix(cm, 'DistilBERT (Test Set)', 'Greens', ax)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(MODEL_DIR), 'cm_distilbert.png'), dpi=150)
    plt.show()

    # Save tokenizer alongside weights
    tokenizer.save_pretrained(MODEL_DIR)
    print(f'\nSaved to {MODEL_DIR}/')

    return get_metrics(y_test, y_pred)


if __name__ == '__main__':
    train()
