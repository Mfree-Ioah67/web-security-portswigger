"""
Data loading and preprocessing utilities for XSS Detection.
"""
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_text(text: str) -> str:
    """
    Normalize raw payload text.

    Steps:
        1. Lowercase and strip whitespace
        2. Collapse multiple spaces
        3. Remove characters that are not alphanumeric, common HTML/JS
           syntax chars, or Vietnamese diacritics
    """
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(
        r'[^\w\s<>/=\'"\-:;\(\)'
        r'\u00e0\u00e1\u1ea1\u1ea3\u00e3\u00e2\u1ea7\u1ea5\u1ead\u1ea9\u1eab'
        r'\u0103\u1eb1\u1eaf\u1eb7\u1eb3\u1eb5'
        r'\u00e8\u00e9\u1eb9\u1ebb\u1ebd\u00ea\u1ec1\u1ebf\u1ec7\u1ec3\u1ec5'
        r'\u00ec\u00ed\u1ecb\u1ec9\u0129'
        r'\u00f2\u00f3\u1ecd\u1ecf\u00f5\u00f4\u1ed3\u1ed1\u1ed9\u1ed5\u1ed7'
        r'\u01a1\u1edf\u1edb\u1ee3\u1edd\u1ee1'
        r'\u00f9\u00fa\u1ee5\u1ee7\u0169\u01b0\u1eeb\u1ee9\u1ef1\u1eed\u1eef'
        r'\u1ef3\u00fd\u1ef5\u1ef7\u1ef9\u0111]', '', text
    )
    return text.strip()


def load_and_split(csv_path: str):
    """
    Load dataset, deduplicate, preprocess, and split 70 / 15 / 15.

    Parameters
    ----------
    csv_path : str
        Path to xss_dataset_500.csv

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  (preprocessed text)
    y_train, y_val, y_test : np.ndarray  (int labels 0/1)
    """
    df = pd.read_csv(csv_path)
    df = df[['payload', 'label']].copy()
    df.columns = ['text', 'label']

    print(f'Raw dataset  : {len(df):,} samples')
    print(f'Duplicates   : {df["text"].duplicated().sum():,}')
    df = df.drop_duplicates(subset='text').reset_index(drop=True)
    print(f'After dedup  : {len(df):,} samples')
    print(f'XSS  (1)     : {(df.label==1).sum():,} ({(df.label==1).mean()*100:.1f}%)')
    print(f'Benign (0)   : {(df.label==0).sum():,} ({(df.label==0).mean()*100:.1f}%)')

    df['clean_text'] = df['text'].apply(preprocess_text)
    X, y = df['clean_text'], df['label']

    # 70 / 15 / 15 stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    print(f'\nTrain        : {len(X_train):>5,} ({len(X_train)/len(X)*100:.1f}%)')
    print(f'Validation   : {len(X_val):>5,} ({len(X_val)/len(X)*100:.1f}%)')
    print(f'Test         : {len(X_test):>5,} ({len(X_test)/len(X)*100:.1f}%)')

    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert len(set(X_val.index)   & set(X_test.index)) == 0
    print('No data leakage between splits.')

    return (
        X_train.values, X_val.values, X_test.values,
        y_train.values, y_val.values, y_test.values
    )
