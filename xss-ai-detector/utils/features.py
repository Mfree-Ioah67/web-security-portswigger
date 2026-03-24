"""
Feature Engineering for XSS Detection.

Handcrafted features extracted from raw payload text.
Used by all 3 models:
  - SVM  : TF-IDF (char n-gram) + handcrafted features (combined sparse matrix)
  - DistilBERT / PhoBERT : handcrafted features appended to [CLS] embedding
                           via a custom classification head (optional),
                           or used standalone for analysis/visualization.

Run standalone to analyze dataset features:
    python utils/features.py
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# XSS pattern list  (32 patterns)
# ---------------------------------------------------------------------------
_XSS_PATTERNS = [
    r'<script',
    r'</script>',
    r'javascript\s*:',
    r'on\w+\s*=',           # onerror=, onload=, onclick=, ...
    r'alert\s*\(',
    r'eval\s*\(',
    r'document\s*\.',
    r'window\s*\.',
    r'<iframe',
    r'<img',
    r'<svg',
    r'<body',
    r'<input',
    r'<form',
    r'src\s*=',
    r'href\s*=',
    r'data\s*:',
    r'vbscript\s*:',
    r'expression\s*\(',
    r'&#x?[0-9a-f]+;',      # HTML entity encoding
    r'%[0-9a-f]{2}',        # URL percent-encoding
    r'\\u[0-9a-f]{4}',      # Unicode escape
    r'fromcharcode',
    r'base64',
    r'atob\s*\(',
    r'fetch\s*\(',
    r'xmlhttprequest',
    r'cookie',
    r'localstorage',
    r'innerhtml',
    r'outerhtml',
    r'write\s*\(',
    r'settimeout\s*\(',
    r'setinterval\s*\(',
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _XSS_PATTERNS]

# Feature names (18 structural + 1 total-pattern count + 34 binary pattern flags)
HANDCRAFTED_FEATURE_NAMES = [
    'log_length',
    'n_html_tags',
    'n_lt', 'n_gt', 'n_eq',
    'n_dquote', 'n_squote',
    'n_lparen', 'n_rparen', 'n_semicolon',
    'n_special_total',
    'ratio_special',
    'n_spaces',
    'n_url_encoding',
    'n_html_entity',
    'n_parens',
    'n_angles',
    'n_xss_patterns_total',
] + [f'pat_{i:02d}_{p[:12].strip("^").replace(chr(92),"").replace(" ","")}' 
     for i, p in enumerate(_XSS_PATTERNS)]


# ---------------------------------------------------------------------------
# 1. Handcrafted feature extractor
# ---------------------------------------------------------------------------
def extract_handcrafted(texts) -> np.ndarray:
    """
    Extract handcrafted features from a list of raw text strings.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features=52), dtype float32

    Feature groups
    --------------
    Structural (18):
        log_length          log(1 + len(text))
        n_html_tags         number of <tag> occurrences
        n_lt/gt/eq/...      counts of special chars: < > = " ' ( ) ;
        n_special_total     total special char count
        ratio_special       n_special_total / length
        n_spaces            whitespace count
        n_url_encoding      count of %xx sequences
        n_html_entity       count of &#...; sequences
        n_parens            ( + ) count
        n_angles            < + > count
        n_xss_patterns_total  total number of XSS patterns matched

    Pattern flags (34 binary):
        pat_00 .. pat_33    1 if the corresponding XSS pattern is found, else 0
    """
    results = []
    for text in texts:
        text  = str(text)
        lower = text.lower()
        length = max(len(text), 1)

        log_len = np.log1p(length)
        n_tags  = len(re.findall(r'<[^>]+>', text))

        sp = {'<': 0, '>': 0, '=': 0, '"': 0, "'": 0, '(': 0, ')': 0, ';': 0}
        for ch in text:
            if ch in sp:
                sp[ch] += 1

        n_special_total = sum(sp.values())
        ratio_special   = n_special_total / length
        n_spaces        = text.count(' ')
        n_url_enc       = len(re.findall(r'%[0-9a-fA-F]{2}', text))
        n_entity        = len(re.findall(r'&#x?[0-9a-fA-F]+;', text))
        n_parens        = sp['('] + sp[')']
        n_angles        = sp['<'] + sp['>']

        pattern_flags   = [1 if p.search(lower) else 0 for p in _COMPILED]
        n_patterns      = sum(pattern_flags)

        row = [
            log_len, n_tags,
            sp['<'], sp['>'], sp['='], sp['"'], sp["'"],
            sp['('], sp[')'], sp[';'],
            n_special_total, ratio_special, n_spaces,
            n_url_enc, n_entity, n_parens, n_angles,
            n_patterns,
        ] + pattern_flags

        results.append(row)

    return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. TF-IDF builder  (used by SVM)
# ---------------------------------------------------------------------------
def build_tfidf(X_train, **kwargs) -> TfidfVectorizer:
    """Fit a character-level TF-IDF vectorizer on the training set."""
    defaults = dict(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=10000,
        sublinear_tf=True,
        strip_accents='unicode',
    )
    defaults.update(kwargs)
    vec = TfidfVectorizer(**defaults)
    vec.fit(X_train)
    return vec


# ---------------------------------------------------------------------------
# 3. Combined features  (TF-IDF sparse + handcrafted dense)  — used by SVM
# ---------------------------------------------------------------------------
def extract_combined(texts, tfidf_vec: TfidfVectorizer):
    """
    Concatenate TF-IDF sparse matrix with handcrafted dense features.

    Returns a sparse matrix of shape
        (n_samples, tfidf_max_features + len(HANDCRAFTED_FEATURE_NAMES))
    """
    tfidf_feats = tfidf_vec.transform(texts)
    hand_feats  = csr_matrix(extract_handcrafted(texts))
    return hstack([tfidf_feats, hand_feats])


# ---------------------------------------------------------------------------
# 4. Transformer-ready features  (used by DistilBERT / PhoBERT)
# ---------------------------------------------------------------------------
def get_transformer_extra_features(texts) -> np.ndarray:
    """
    Return normalized handcrafted features for transformer models.
    These can be concatenated with the [CLS] token embedding before
    the final classification layer.

    Returns
    -------
    np.ndarray of shape (n_samples, 52), dtype float32, values in [0, 1]
    """
    feats = extract_handcrafted(texts)
    # Min-max normalize each feature column
    col_min = feats.min(axis=0)
    col_max = feats.max(axis=0)
    denom   = np.where(col_max - col_min == 0, 1, col_max - col_min)
    return (feats - col_min) / denom


# ---------------------------------------------------------------------------
# 5. Standalone analysis
# ---------------------------------------------------------------------------
def analyze_features(csv_path: str):
    """Print feature statistics and save visualization plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(csv_path)
    df = df[['payload', 'label']].dropna()
    df.columns = ['text', 'label']
    df = df.drop_duplicates(subset='text').reset_index(drop=True)

    print(f'Dataset : {len(df):,} samples')
    print(f'XSS (1) : {(df.label==1).sum():,}')
    print(f'Benign  : {(df.label==0).sum():,}')

    feats    = extract_handcrafted(df['text'].values)
    feat_df  = pd.DataFrame(feats, columns=HANDCRAFTED_FEATURE_NAMES)
    feat_df['label'] = df['label'].values

    xss_mean    = feat_df[feat_df.label == 1].drop('label', axis=1).mean()
    benign_mean = feat_df[feat_df.label == 0].drop('label', axis=1).mean()
    diff = (xss_mean - benign_mean).abs().sort_values(ascending=False)

    print('\nTop 10 discriminative features (|mean_XSS - mean_Benign|):')
    print(diff.head(10).to_string())

    out_dir = os.path.join(os.path.dirname(csv_path), '..', 'models')
    os.makedirs(out_dir, exist_ok=True)

    # Bar chart — top 10
    fig, ax = plt.subplots(figsize=(11, 5))
    diff.head(10).plot(kind='bar', ax=ax, color='#e74c3c',
                       edgecolor='black', alpha=0.85)
    ax.set_title('Top 10 Handcrafted Features  |Mean XSS - Mean Benign|',
                 fontweight='bold')
    ax.set_ylabel('Absolute Mean Difference')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
    plt.tight_layout()
    p1 = os.path.normpath(os.path.join(out_dir, 'feature_analysis.png'))
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {p1}')
    plt.show()

    # Correlation heatmap — top 10 features + label
    top_cols = list(diff.head(10).index) + ['label']
    corr = feat_df[top_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax2, annot_kws={'fontsize': 9})
    ax2.set_title('Feature Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    p2 = os.path.normpath(os.path.join(out_dir, 'feature_correlation.png'))
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    print(f'Saved: {p2}')
    plt.show()


def list_all_features():
    """
    Print a full listing of every feature used in the XSS Detector.

    Groups
    ------
    A. Dataset columns  (raw CSV)
    B. Handcrafted features  (extracted from payload text)
       B1. Structural  (18)
       B2. XSS pattern flags  (34 binary)
    C. TF-IDF character n-gram features  (SVM only, 10 000)
    """
    SEP = '=' * 65

    print(SEP)
    print('GIAI THICH DAC TRUNG (FEATURE EXPLANATION - TIENG VIET)')
    print(SEP)
    print()
    print('Cac dac trung duoc chia lam 3 nhom chinh:')
    print()
    print('  A. Cot du lieu goc (Dataset Columns)')
    print('     - Day la cac cot co san trong file CSV, khong can tinh toan.')
    print('     - "payload" : chuoi van ban dau vao can phan tich')
    print('     - "label"   : nhan du doan (0 = Binh thuong, 1 = XSS tan cong)')
    print('     - "type"    : loai payload (vi du: script_tag, img_tag, benign...)')
    print('     - "source"  : nguon du lieu (OWASP, GitHub, wafpass...)')
    print()
    print('  B. Dac trung thu cong (Handcrafted Features) - 52 dac trung')
    print('     - Duoc TINH TOAN tu noi dung cua cot "payload".')
    print('     - Giup mo hinh hoc cac dau hieu dac trung cua XSS ma mat nguoi co the nhan ra.')
    print()
    print('     B1. Dac trung cau truc (Structural) - 18 dac trung:')
    print('         Phan tich hinh thuc cua chuoi van ban:')
    print('         - Do dai chuoi (log_length)')
    print('         - So luong the HTML nhu <script>, <img>, <iframe>...')
    print('         - So luong ky tu dac biet: < > = " ( ) ;')
    print('         - Ti le ky tu dac biet tren tong do dai')
    print('         - So luong ma hoa URL (%xx) va HTML entity (&#...;)')
    print('         - Tong so pattern XSS khop duoc')
    print()
    print('     B2. Co hieu XSS (XSS Pattern Flags) - 34 dac trung nhi phan (0/1):')
    print('         Moi dac trung la 1 neu payload chua pattern nguy hiem, 0 neu khong:')
    print('         - <script, </script>  : the script nhung ma doc')
    print('         - javascript:         : giao thuc javascript trong href/src')
    print('         - on*= (onerror=...)  : su kien HTML de chay JS')
    print('         - alert(, eval(       : ham JS thuong dung trong XSS')
    print('         - document., window.  : truy cap DOM/window')
    print('         - base64, atob(       : ma hoa de an payload')
    print('         - cookie, localStorage: danh cap du lieu nguoi dung')
    print('         - innerHTML, write(   : ghi noi dung vao trang')
    print('         - setTimeout/Interval : chay JS bi tri hoan')
    print()
    print('  C. Dac trung TF-IDF (chi dung cho SVM) - 10 000 dac trung:')
    print('     - Phan tich cac cum ky tu (n-gram) cap do ky tu (2-5 ky tu).')
    print('     - Vi du: "<sc", "scr", "cri", "ipt>", "alert", "onerr"...')
    print('     - Giup SVM bat duoc cac pattern XSS bi lam mo (obfuscation).')
    print('     - Ket hop voi 52 dac trung thu cong -> tong 10 052 dac trung cho SVM.')
    print()
    print('  => Voi DistilBERT va PhoBERT:')
    print('     Mo hinh doc hieu ngu nghia cua payload qua co che Attention,')
    print('     sau do ket hop voi 52 dac trung thu cong truoc khi phan loai.')
    print('     Kien truc: [CLS] embedding (768 chieu) + 52 dac trung -> 256 -> 2 lop')
    print()
    print(SEP)

    # ── A. Dataset columns ────────────────────────────────────────
    print(SEP)
    print('A. DATASET COLUMNS  (xss_dataset_500.csv)')
    print(SEP)
    dataset_cols = [
        ('payload', 'Raw text / XSS payload string'),
        ('label',   '0 = Benign, 1 = XSS  (prediction target)'),
        ('type',    'Payload category: script_tag, img_tag, benign, ...'),
        ('source',  'Data origin: OWASP, GitHub, wafpass, ...'),
    ]
    for col, desc in dataset_cols:
        print(f'  {col:<12}  {desc}')

    # ── B1. Structural handcrafted ────────────────────────────────
    print()
    print(SEP)
    print('B. HANDCRAFTED FEATURES  (extracted from payload)')
    print(SEP)
    print()
    print('  B1. Structural features  (18)')
    print('  ' + '-'*61)
    structural = [
        ('log_length',          'log(1 + len(payload))'),
        ('n_html_tags',         'Number of <tag> occurrences'),
        ('n_lt',                'Count of "<"'),
        ('n_gt',                'Count of ">"'),
        ('n_eq',                'Count of "="'),
        ('n_dquote',            'Count of \'"\' (double quote)'),
        ('n_squote',            "Count of \"'\" (single quote)"),
        ('n_lparen',            'Count of "("'),
        ('n_rparen',            'Count of ")"'),
        ('n_semicolon',         'Count of ";"'),
        ('n_special_total',     'Total special character count'),
        ('ratio_special',       'n_special_total / len(payload)'),
        ('n_spaces',            'Whitespace count'),
        ('n_url_encoding',      'Count of %xx sequences'),
        ('n_html_entity',       'Count of &#...; sequences'),
        ('n_parens',            'n_lparen + n_rparen'),
        ('n_angles',            'n_lt + n_gt'),
        ('n_xss_patterns_total','Total number of XSS patterns matched'),
    ]
    for i, (name, desc) in enumerate(structural, 1):
        print(f'  {i:>2}. {name:<28} {desc}')

    # ── B2. XSS pattern flags ─────────────────────────────────────
    print()
    print('  B2. XSS pattern flags  (34 binary: 1 = pattern found, 0 = not)')
    print('  ' + '-'*61)
    pattern_descs = [
        '<script',          '</script>',        'javascript:',
        'on*= (onerror=, onload=, onclick=...)', 'alert(',
        'eval(',            'document.',         'window.',
        '<iframe',          '<img',              '<svg',
        '<body',            '<input',            '<form',
        'src=',             'href=',             'data:',
        'vbscript:',        'expression(',       '&#x...; (HTML entity encoding)',
        '%xx (URL percent-encoding)',             r'\uXXXX (Unicode escape)',
        'fromcharcode',     'base64',            'atob(',
        'fetch(',           'xmlhttprequest',    'cookie',
        'localstorage',     'innerHTML',         'outerHTML',
        'write(',           'setTimeout(',       'setInterval(',
    ]
    for i, (name, desc) in enumerate(zip(HANDCRAFTED_FEATURE_NAMES[18:], pattern_descs)):
        print(f'  {i+19:>2}. {name:<28} matches: {desc}')

    print()
    print(f'  Total handcrafted features : {len(HANDCRAFTED_FEATURE_NAMES)}')

    # ── C. TF-IDF ─────────────────────────────────────────────────
    print()
    print(SEP)
    print('C. TF-IDF CHARACTER N-GRAM FEATURES  (SVM only)')
    print(SEP)
    print('  analyzer     : char')
    print('  ngram_range  : (2, 5)  — bigrams to 5-grams')
    print('  max_features : 10 000')
    print('  sublinear_tf : True')
    print('  Examples     : "<sc", "scr", "cri", "rip", "ipt>",')
    print('                 "alert", "aler", "onerr", "eval("')
    print()
    print('  SVM total feature vector size: 10 000 (TF-IDF) + 52 (handcrafted)')
    print(f'                               = 10 052 features')

    # ── Summary ───────────────────────────────────────────────────
    print()
    print(SEP)
    print('SUMMARY')
    print(SEP)
    rows = [
        ('SVM',        'TF-IDF (10 000) + Handcrafted (52)',  '10 052'),
        ('DistilBERT', '[CLS] (768) + Handcrafted (52)', '820 -> 256 -> 2'),
        ('PhoBERT',    '[CLS] (768) + Handcrafted (52)', '820 -> 256 -> 2'),
    ]
    print(f'  {"Model":<12} {"Features":<44} {"Classifier input"}')
    print('  ' + '-'*61)
    for model, feats, clf in rows:
        print(f'  {model:<12} {feats:<44} {clf}')
    print(SEP)


if __name__ == '__main__':
    list_all_features()
    print()
    csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'data', 'xss_dataset_500.csv')
    analyze_features(csv)
