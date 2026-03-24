"""
Evaluation utilities shared across all three models.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)


def get_metrics(y_true, y_pred) -> dict:
    """Return accuracy, precision, recall, f1 as a dict."""
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }


def compute_metrics_hf(eval_pred):
    """Callback for HuggingFace Trainer.compute_metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return get_metrics(labels, preds)


def print_report(y_true, y_pred, model_name: str):
    """Print a formatted classification report."""
    print(f'\n{"="*55}')
    print(f'Results  —  {model_name}  (Test Set)')
    print('='*55)
    print(classification_report(y_true, y_pred,
                                 target_names=['Benign', 'XSS'], digits=4))


def plot_confusion_matrix(cm, title: str, cmap: str, ax=None):
    """Draw a single confusion matrix on the given axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Benign', 'XSS'],
                yticklabels=['Benign', 'XSS'],
                annot_kws={'fontsize': 13, 'fontweight': 'bold'})
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f = 2 * p * r / (p + r) if (p + r) else 0
    ax.set_title(f'{title}\nP={p:.3f}  R={r:.3f}  F1={f:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


def plot_comparison(metrics_dict: dict, save_path: str = None):
    """
    Bar chart comparing Accuracy / Precision / Recall / F1 across models.

    Parameters
    ----------
    metrics_dict : {'ModelName': {'accuracy': ..., 'precision': ..., ...}, ...}
    save_path    : optional file path to save the figure
    """
    models  = list(metrics_dict.keys())
    keys    = ['accuracy', 'precision', 'recall', 'f1']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors  = ['#3498db', '#2ecc71', '#e74c3c']

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('XSS Detector — Model Comparison (Test Set)',
                 fontsize=15, fontweight='bold')

    for idx, (mkey, mlabel) in enumerate(zip(keys, labels)):
        ax   = axes[idx // 2, idx % 2]
        vals = [metrics_dict[m][mkey] for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)],
                      alpha=0.85, edgecolor='black')
        ax.set_title(mlabel, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Score')
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()
