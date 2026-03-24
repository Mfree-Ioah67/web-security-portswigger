"""
Train SVM + TF-IDF + Handcrafted Features for XSS detection.

Features used:
    - Character-level TF-IDF (2-5 grams, 10 000 features)
    - 52 handcrafted features (structural + XSS pattern flags)

Run:
    python training/train_svm.py
"""
import os
import sys
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocess import load_and_split
from utils.evaluate import get_metrics, print_report, plot_confusion_matrix
from utils.features import build_tfidf, extract_combined

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'xss_dataset_500.csv')
MODEL_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'models')


def train():
    print('=' * 55)
    print('Training  SVM + TF-IDF + Handcrafted Features')
    print('=' * 55)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATASET_PATH)

    # --- Feature extraction ---
    print('\nExtracting features...')
    tfidf        = build_tfidf(X_train)
    X_train_feat = extract_combined(X_train, tfidf)
    X_val_feat   = extract_combined(X_val,   tfidf)
    X_test_feat  = extract_combined(X_test,  tfidf)
    print(f'Feature matrix shape (train): {X_train_feat.shape}')
    print(f'  TF-IDF features : 10 000')
    print(f'  Handcrafted     : 52  (18 structural + 34 XSS pattern flags)')

    # --- Train ---
    start = time.time()
    svm   = SVC(kernel='linear', probability=True, random_state=42, C=1.0)
    svm.fit(X_train_feat, y_train)
    elapsed = time.time() - start

    # --- Evaluate ---
    y_pred_val  = svm.predict(X_val_feat)
    y_pred_test = svm.predict(X_test_feat)
    acc_val     = accuracy_score(y_val,  y_pred_val)
    acc_test    = accuracy_score(y_test, y_pred_test)
    gap         = abs(acc_val - acc_test)

    print(f'\nTraining time  : {elapsed:.2f}s')
    print(f'Val  Accuracy  : {acc_val:.4f}')
    print(f'Test Accuracy  : {acc_test:.4f}')
    print(f'Val-Test gap   : {gap:.4f}  {"OK" if gap < 0.03 else "Possible overfit"}')

    print_report(y_test, y_pred_test, 'SVM + TF-IDF + Handcrafted')

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion_matrix(cm, 'SVM (Test Set)', 'Blues', ax)
    plt.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(os.path.join(MODEL_DIR, 'cm_svm.png'), dpi=150)
    plt.show()

    # --- Save ---
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(svm,   os.path.join(MODEL_DIR, 'svm_xss_model.pkl'))
    print(f'\nSaved to {MODEL_DIR}/')

    return get_metrics(y_test, y_pred_test)


if __name__ == '__main__':
    train()
