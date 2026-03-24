# XSS AI Detector

Detect Cross-Site Scripting (XSS) attacks using three machine learning models:
**SVM + TF-IDF**, **DistilBERT**, and **PhoBERT** — with a Flask web interface.

---

## Project Structure

```
xss-ai-detector/
├── api/                        # Flask web service
│   ├── app.py                  # API server + routes
│   └── templates/index.html    # Web UI
├── data/
│   └── xss_dataset_500.csv     # Dataset (500 samples, labels: 0=Benign, 1=XSS)
├── models/                     # Trained model weights & outputs
│   ├── distilbert/             # Fine-tuned DistilBERT
│   ├── phobert/                # Fine-tuned PhoBERT
│   ├── svm_xss_model.pkl       # Trained SVM model
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│   └── *.png                   # Evaluation charts
├── training/
│   ├── train_svm.py            # Train SVM + TF-IDF + Handcrafted features
│   ├── train_distilbert.py     # Train DistilBERT + Handcrafted features
│   └── train_phobert.py        # Train PhoBERT + Handcrafted features
├── utils/
│   ├── features.py             # Feature engineering (52 handcrafted features)
│   ├── preprocess.py           # Text preprocessing + train/val/test split
│   └── evaluate.py             # Metrics, reports, confusion matrix plots
├── XSS_Detector.ipynb          # Google Colab notebook version
├── compare_models.py           # Evaluate & compare all 3 models
├── run.py                      # Entry point (train / serve)
└── requirements.txt
```

---

## Dataset

File: `data/xss_dataset_500.csv`

| Column  | Description |
|---------|-------------|
| payload | Raw input string (XSS payload or normal text) |
| label   | 0 = Benign, 1 = XSS |
| type    | Payload category (script_tag, img_tag, benign, ...) |
| source  | Data origin (OWASP, GitHub, wafpass, ...) |

Split strategy: **70% train / 15% val / 15% test** (stratified, no data leakage)

---

## Features

### Handcrafted Features (52) — used by all 3 models

**Structural (18):**

| Feature | Description |
|---------|-------------|
| log_length | log(1 + payload length) |
| n_html_tags | Number of HTML tags |
| n_lt / n_gt / n_eq | Count of `<` `>` `=` |
| n_dquote / n_squote | Count of `"` `'` |
| n_lparen / n_rparen / n_semicolon | Count of `(` `)` `;` |
| n_special_total | Total special character count |
| ratio_special | Special chars / total length |
| n_spaces | Whitespace count |
| n_url_encoding | Count of `%xx` sequences |
| n_html_entity | Count of `&#...;` sequences |
| n_parens / n_angles | Paired bracket counts |
| n_xss_patterns_total | Total XSS patterns matched |

**XSS Pattern Flags (34 binary):**
`<script`, `</script>`, `javascript:`, `on*=`, `alert(`, `eval(`, `document.`,
`window.`, `<iframe`, `<img`, `<svg`, `<body`, `<input`, `<form`, `src=`, `href=`,
`data:`, `vbscript:`, `expression(`, HTML entity encoding, URL encoding,
Unicode escape, `fromcharcode`, `base64`, `atob(`, `fetch(`, `xmlhttprequest`,
`cookie`, `localStorage`, `innerHTML`, `outerHTML`, `write(`, `setTimeout(`, `setInterval(`

### TF-IDF Features (10,000) — SVM only
Character-level n-grams (2–5), e.g. `<sc`, `scr`, `alert`, `onerr`
Combined with handcrafted → **10,052 total features** for SVM.

### Transformer Features — DistilBERT & PhoBERT
`[CLS] embedding (768)` + `Handcrafted (52)` → `Linear(820→256)` → `Linear(256→2)`

---

## Models & Results

| Model | Accuracy | Precision | Recall | F1-Score | Infer (ms/sample) |
|-------|----------|-----------|--------|----------|-------------------|
| SVM + TF-IDF + Handcrafted | **98.36%** | 1.0000 | 0.9730 | **0.9863** | 0.05 |
| PhoBERT + Handcrafted | 96.72% | 0.9730 | 0.9730 | 0.9730 | 223.67 |
| DistilBERT + Handcrafted | 83.61% | 0.8462 | 0.8919 | 0.8684 | 49.70 |

> DistilBERT result is from pretrained base weights (not yet fine-tuned on this dataset).
> Re-train with `python run.py train distilbert` to improve.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Train models

```bash
python run.py train svm          # ~1 second
python run.py train distilbert   # GPU recommended
python run.py train phobert      # GPU recommended
python run.py train all          # Train all 3
```

### Compare all models

```bash
python compare_models.py
```

### Run web server

```bash
python run.py serve
# Local:  http://localhost:5000
# Public: ngrok URL printed in terminal
```

### Analyze features

```bash
python utils/features.py
```

---

## API

**POST** `/predict`

```json
{
  "payload": "<script>alert(1)</script>",
  "model": "svm"
}
```

Response:
```json
{
  "model": "svm",
  "label": 1,
  "is_xss": true,
  "verdict": "XSS Detected",
  "confidence": 0.9999,
  "prob_benign": 0.0001,
  "prob_xss": 0.9999
}
```

`model` options: `svm` | `distilbert` | `phobert`

---

## Google Colab

Open `XSS_Detector.ipynb` in Google Colab with T4 GPU for full training.
