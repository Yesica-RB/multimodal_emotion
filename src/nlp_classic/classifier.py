# src/nlp_classic/classifier.py
# Classical NLP classification module.
# Uses TF-IDF vectorisation with two classifiers:
#   1. Multinomial Naive Bayes  → fast baseline
#   2. Logistic Regression      → stronger discriminative model
#
# Both models are evaluated with macro-averaged F1 to handle class imbalance.
# Probabilities are saved for the late fusion module (Week 4).

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from .preprocessing import TextPreprocessor


def run_nlp_classic(df,
                    text_col:  str = 'text',
                    label_col: str = 'label',
                    save_path: str = 'results/metrics_nlp_classic.json'):
    """Train and evaluate Naive Bayes and Logistic Regression on tweet captions.

    Args:
        df:        DataFrame with text and label columns.
        text_col:  Name of the column containing tweet captions.
        label_col: Name of the column containing emotion labels.
        save_path: Path to save the JSON results file.

    Returns:
        nb, lr, vec, y_test — trained models and test labels.
    """
    os.makedirs('results', exist_ok=True)

    # ── Preprocessing ────────────────────────────────────────────
    prep   = TextPreprocessor()
    print("Preprocessing texts...")
    texts  = [prep.preprocess(t) for t in df[text_col]]
    labels = df[label_col].tolist()

    # 80/20 train-test split with stratification to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2, random_state=42, stratify=labels)

    # ── TF-IDF Vectorisation ─────────────────────────────────────
    # max_features=10000 keeps the 10k most informative terms
    # ngram_range=(1,2) includes unigrams and bigrams
    # sublinear_tf=True applies log normalisation to term frequency
    vec  = TfidfVectorizer(max_features=10000,
                           ngram_range=(1, 2),
                           sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    results = {}

    # ── Naive Bayes ──────────────────────────────────────────────
    # alpha=0.1 is the Laplace smoothing parameter (avoids zero probabilities)
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_tr, y_train)
    y_nb = nb.predict(X_te)

    print("\n=== Naive Bayes ===")
    print(classification_report(y_test, y_nb))

    results['naive_bayes'] = {
        'f1_macro': f1_score(y_test, y_nb, average='macro'),
        'report':   classification_report(y_test, y_nb, output_dict=True)
    }

    # ── Logistic Regression ──────────────────────────────────────
    # C=1.0 controls L2 regularisation strength (higher = less regularisation)
    # max_iter=1000 ensures convergence on this dataset size
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_tr, y_train)
    y_lr = lr.predict(X_te)

    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_lr))

    results['logistic_regression'] = {
        'f1_macro': f1_score(y_test, y_lr, average='macro'),
        'report':   classification_report(y_test, y_lr, output_dict=True),
        'probas':   lr.predict_proba(X_te).tolist()  # saved for late fusion
    }

    # ── Save results ─────────────────────────────────────────────
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    return nb, lr, vec, y_test
