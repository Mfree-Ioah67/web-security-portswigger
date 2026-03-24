"""
Evaluate and compare all three models on the held-out test set.

Run:
    python compare_models.py
"""
import os, sys, time
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocess import load_and_split
from utils.features import build_tfidf, extract_combined, get_transformer_extra_features
from utils.evaluate import get_metrics, print_report, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATASET = os.path.join('data', 'xss_dataset_500.csv')
MDL_DIR = 'models'

# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATASET)

results = {}   # model_name -> metrics
cms     = {}   # model_name -> confusion matrix
times   = {}   # model_name -> ms / sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')

# ---------------------------------------------------------------------------
# 1. SVM  (TF-IDF + Handcrafted)
# ---------------------------------------------------------------------------
print('\n' + '='*55)
print('Evaluating  SVM + TF-IDF + Handcrafted')
print('='*55)

tfidf = joblib.load(os.path.join(MDL_DIR, 'tfidf_vectorizer.pkl'))
svm   = joblib.load(os.path.join(MDL_DIR, 'svm_xss_model.pkl'))
X_test_feat = extract_combined(X_test, tfidf)

t0 = time.time()
y_pred_svm = svm.predict(X_test_feat)
times['SVM'] = (time.time() - t0) / len(X_test) * 1000

results['SVM'] = get_metrics(y_test, y_pred_svm)
cms['SVM']     = confusion_matrix(y_test, y_pred_svm)
print(classification_report(y_test, y_pred_svm,
      target_names=['Benign', 'XSS'], digits=4))

# ---------------------------------------------------------------------------
# 2. DistilBERT  (DistilBERT [CLS] + Handcrafted)
# ---------------------------------------------------------------------------
print('\n' + '='*55)
print('Evaluating  DistilBERT + Handcrafted')
print('='*55)

from transformers import DistilBertTokenizerFast
from training.train_distilbert import DistilBertWithFeatures, XSSDataset as DistilDS

distil_path = os.path.join(MDL_DIR, 'distilbert')
try:
    tok_d = DistilBertTokenizerFast.from_pretrained(distil_path)
except Exception:
    tok_d = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

extra_test_d = get_transformer_extra_features(X_test)
test_ds_d    = DistilDS(X_test, y_test, tok_d, extra_test_d)
loader_d     = DataLoader(test_ds_d, batch_size=32)

model_d = DistilBertWithFeatures().to(device)
ckpt_d  = os.path.join(distil_path, 'best_model.pt')
if os.path.exists(ckpt_d):
    model_d.load_state_dict(torch.load(ckpt_d, map_location=device))
    print(f'Loaded weights from {ckpt_d}')
else:
    print(f'Warning: {ckpt_d} not found — using untrained weights.')
model_d.eval()

preds_d = []
t0 = time.time()
with torch.no_grad():
    for batch in loader_d:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        ext  = batch['extra'].to(device)
        out  = model_d(ids, mask, ext)
        preds_d.extend(out.argmax(-1).cpu().tolist())
times['DistilBERT'] = (time.time() - t0) / len(X_test) * 1000

y_pred_distil = np.array(preds_d)
results['DistilBERT'] = get_metrics(y_test, y_pred_distil)
cms['DistilBERT']     = confusion_matrix(y_test, y_pred_distil)
print(classification_report(y_test, y_pred_distil,
      target_names=['Benign', 'XSS'], digits=4))

# ---------------------------------------------------------------------------
# 3. PhoBERT  (PhoBERT [CLS] + Handcrafted)
# ---------------------------------------------------------------------------
print('\n' + '='*55)
print('Evaluating  PhoBERT + Handcrafted')
print('='*55)

from transformers import AutoTokenizer
from training.train_phobert import PhoBertWithFeatures, XSSDataset as PhoDS

pho_path = os.path.join(MDL_DIR, 'phobert')
tok_p    = AutoTokenizer.from_pretrained(pho_path)

extra_test_p = get_transformer_extra_features(X_test)
test_ds_p    = PhoDS(X_test, y_test, tok_p, extra_test_p)
loader_p     = DataLoader(test_ds_p, batch_size=16)

model_p = PhoBertWithFeatures().to(device)
ckpt_p  = os.path.join(pho_path, 'best_model.pt')
if os.path.exists(ckpt_p):
    model_p.load_state_dict(torch.load(ckpt_p, map_location=device))
    print(f'Loaded weights from {ckpt_p}')
else:
    print(f'Warning: {ckpt_p} not found — using pretrained base weights only.')
model_p.eval()

preds_p = []
t0 = time.time()
with torch.no_grad():
    for batch in loader_p:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        ext  = batch['extra'].to(device)
        out  = model_p(ids, mask, ext)
        preds_p.extend(out.argmax(-1).cpu().tolist())
times['PhoBERT'] = (time.time() - t0) / len(X_test) * 1000

y_pred_pho = np.array(preds_p)
results['PhoBERT'] = get_metrics(y_test, y_pred_pho)
cms['PhoBERT']     = confusion_matrix(y_test, y_pred_pho)
print(classification_report(y_test, y_pred_pho,
      target_names=['Benign', 'XSS'], digits=4))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print('\n' + '='*68)
print('MODEL COMPARISON  —  Test Set')
print('='*68)
rows = []
for name, m in results.items():
    rows.append({
        'Model':            name,
        'Accuracy':         m['accuracy'],
        'Precision':        m['precision'],
        'Recall':           m['recall'],
        'F1-Score':         m['f1'],
        'Infer (ms/sample)': round(times[name], 3),
    })
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format='{:.4f}'.format))
print('='*68)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
sns.set_style('whitegrid')
model_names   = list(results.keys())
metric_keys   = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors        = ['#3498db', '#2ecc71', '#e74c3c']

# Bar chart — 4 metrics
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('XSS Detector — Model Comparison (Test Set)',
             fontsize=15, fontweight='bold')
for idx, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
    ax   = axes[idx // 2, idx % 2]
    vals = [results[m][mkey] for m in model_names]
    bars = ax.bar(model_names, vals, color=colors, alpha=0.85, edgecolor='black')
    ax.set_title(mlabel, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
plt.tight_layout()
p1 = os.path.join(MDL_DIR, 'model_comparison.png')
plt.savefig(p1, dpi=200, bbox_inches='tight')
print(f'\nSaved: {p1}')
plt.show()

# Confusion matrices
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
fig2.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold')
for i, (name, cmap) in enumerate([
    ('SVM', 'Blues'), ('DistilBERT', 'Greens'), ('PhoBERT', 'Oranges')
]):
    plot_confusion_matrix(cms[name], name, cmap, axes2[i])
plt.tight_layout()
p2 = os.path.join(MDL_DIR, 'confusion_matrices.png')
plt.savefig(p2, dpi=200, bbox_inches='tight')
print(f'Saved: {p2}')
plt.show()

# Inference time
fig3, ax3 = plt.subplots(figsize=(7, 4))
t_vals = [times[m] for m in model_names]
bars3  = ax3.bar(model_names, t_vals, color=colors, alpha=0.85, edgecolor='black')
ax3.set_title('Inference Time (ms / sample)', fontweight='bold')
ax3.set_ylabel('ms')
for bar, val in zip(bars3, t_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.3f}ms', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(MDL_DIR, 'inference_time.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
print(f'Saved: {p3}')
plt.show()

print('\nComparison complete.')
