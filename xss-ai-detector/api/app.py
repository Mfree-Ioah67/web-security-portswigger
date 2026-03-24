"""
Flask Web Service cho XSS Detector.
Chạy: python api/app.py
Truy cập: http://localhost:5000
"""
import os
import sys
import joblib
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocess import preprocess_text
from utils.features import extract_combined

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
TMPL_DIR   = os.path.join(os.path.dirname(__file__), 'templates')

app = Flask(__name__, template_folder=TMPL_DIR)

# ── Lazy-load models ──────────────────────────────────────────────
_models = {}


def _load_svm():
    if 'svm' not in _models:
        tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
        svm   = joblib.load(os.path.join(MODEL_DIR, 'svm_xss_model.pkl'))
        _models['svm'] = (tfidf, svm)
        print('✅ SVM loaded')
    return _models['svm']


def _load_distilbert():
    if 'distilbert' not in _models:
        from transformers import (DistilBertTokenizerFast,
                                   DistilBertForSequenceClassification)
        path = os.path.join(MODEL_DIR, 'distilbert')
        # Load tokenizer từ pretrained nếu local thiếu vocab
        try:
            tok = DistilBertTokenizerFast.from_pretrained(path)
        except Exception:
            tok = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(path)
        model.eval()
        _models['distilbert'] = (tok, model)
        print('✅ DistilBERT loaded')
    return _models['distilbert']


def _load_phobert():
    if 'phobert' not in _models:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = os.path.join(MODEL_DIR, 'phobert')
        tok   = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        _models['phobert'] = (tok, model)
        print('✅ PhoBERT loaded')
    return _models['phobert']


# ── Predict helpers ───────────────────────────────────────────────

def predict_svm(text: str) -> dict:
    tfidf, svm = _load_svm()
    clean = preprocess_text(text)
    vec   = extract_combined([clean], tfidf)   # TF-IDF + handcrafted
    label = int(svm.predict(vec)[0])
    proba = svm.predict_proba(vec)[0]
    return {'label': label, 'confidence': float(proba[label]),
            'prob_benign': float(proba[0]), 'prob_xss': float(proba[1])}


def predict_transformer(text: str, loader_fn) -> dict:
    tok, model = loader_fn()
    clean = preprocess_text(text)
    enc = tok(clean, return_tensors='pt', truncation=True,
               padding='max_length', max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
    proba = torch.softmax(logits, dim=-1)[0].tolist()
    label = int(np.argmax(proba))
    return {'label': label, 'confidence': float(proba[label]),
            'prob_benign': float(proba[0]), 'prob_xss': float(proba[1])}


# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json(force=True)
    payload = data.get('payload', '').strip()
    model   = data.get('model', 'svm').lower()

    if not payload:
        return jsonify({'error': 'payload is required'}), 400

    try:
        if model == 'svm':
            result = predict_svm(payload)
        elif model == 'distilbert':
            result = predict_transformer(payload, _load_distilbert)
        elif model == 'phobert':
            result = predict_transformer(payload, _load_phobert)
        else:
            return jsonify({'error': f'Unknown model: {model}'}), 400

        result['model']   = model
        result['is_xss']  = result['label'] == 1
        result['verdict'] = '⚠️ XSS Detected' if result['is_xss'] else '✅ Benign'
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import threading

    NGROK_TOKEN = '3AtkEdL03tK0yA4jmDxzBOM5wug_597AEFVu4CPjsaAwnopMq'

    from pyngrok import ngrok, conf
    conf.get_default().auth_token = NGROK_TOKEN
    ngrok.kill()  # tắt tunnel cũ nếu có

    public_url = ngrok.connect(5000)
    print('\n' + '=' * 55)
    print('🚀 XSS Detector API đang chạy!')
    print(f'🌐 Public URL : {public_url}')
    print('📍 Local URL  : http://localhost:5000')
    print('=' * 55)
    print('   Models load on first request (lazy loading)')
    print('   Ctrl+C để dừng server\n')

    app.run(host='0.0.0.0', port=5000, debug=False)
