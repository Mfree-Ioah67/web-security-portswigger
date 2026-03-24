# IAW301 - Information Assurance & Web Security

> Hands-on repository for **web application security** — vulnerability analysis, ethical hacking labs, and an AI-powered XSS detection system.
> FPT University — Group 5

---

## Repository Structure

```
IAW301/
├── Hands-on lab/          # Lab reports (Group 5 submissions)
├── Material/
│   ├── Slide/             # Course slides (Chapter 1-11)
│   └── Student_Lab/       # Lab instruction documents
├── Report/                # Final project report + XSS dataset
├── xss-ai-detector/       # AI-based XSS Detection System
└── Web Application Pentesting Roadmap.pdf
```

---

## Hands-on Labs

| Lab | Topic |
|-----|-------|
| Lab 1  | Information Disclosure |
| Lab 2  | Authentication — Password-based |
| Lab 3  | Authentication — Multi-factor |
| Lab 4  | Authentication — Other mechanisms |
| Lab 5  | Access Control — IDOR |
| Lab 6  | Access Control — Privilege Escalation |
| Lab 7  | Access Control — OAuth 2.0 |
| Lab 8  | SQL Injection — UNION Attacks |
| Lab 9  | Blind SQL Injection — Time-based |
| Lab 10 | Blind SQL Injection — Error-based |
| Lab 11 | OS Command Injection |
| Lab 12 | Blind OS Command Injection |
| Lab 13 | Cross-Site Scripting — Reflected |

---

## XSS AI Detector

An AI system that detects XSS attacks using three models with handcrafted feature engineering.

### Models

| Model | Accuracy | F1-Score | Speed |
|-------|----------|----------|-------|
| SVM + TF-IDF + Handcrafted | **98.36%** | **0.9863** | 0.05ms/sample |
| PhoBERT + Handcrafted | 96.72% | 0.9730 | 223ms/sample |
| DistilBERT + Handcrafted | 83.61% | 0.8684 | 50ms/sample |

### Features (52 handcrafted + 10,000 TF-IDF for SVM)
- Structural: payload length, HTML tag count, special char ratios, URL/entity encoding
- XSS pattern flags: `<script`, `onerror=`, `alert(`, `eval(`, `base64`, `cookie`, `innerHTML`...

### Quick Start

```bash
cd xss-ai-detector
pip install -r requirements.txt

python run.py train svm        # Train SVM (~1s)
python run.py serve            # Start web server → http://localhost:5000
python compare_models.py       # Compare all 3 models
```

Full documentation: [xss-ai-detector/README.md](xss-ai-detector/README.md)

---

## Covered Topics

- Information Disclosure
- Authentication Vulnerabilities (Password, MFA, OAuth 2.0)
- Access Control (IDOR, Privilege Escalation)
- SQL Injection (UNION, Blind Time-based, Blind Error-based)
- OS Command Injection (Direct & Blind)
- Cross-Site Scripting (XSS) — Reflected
- AI/ML-based XSS Detection

---

## Tools & Platforms

- Burp Suite
- sqlmap
- PortSwigger Web Security Academy
- Python / scikit-learn / PyTorch / HuggingFace Transformers
- Flask

---

## References

- [PortSwigger Web Security Academy](https://portswigger.net/web-security/all-topics)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PhoBERT — VinAI Research](https://github.com/VinAIResearch/PhoBERT)

---

*FPT University — IAW301 — Group 5*
