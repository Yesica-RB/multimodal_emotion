# src/nlp_classic/classifier.py
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from .preprocessing import TextPreprocessor

def run_nlp_classic(df, text_col='text', label_col='label',
                    save_path='results/metrics_nlp_classic.json'):
    os.makedirs('results', exist_ok=True)
    prep = TextPreprocessor()

    print("Preprocesando textos...")
    texts  = [prep.preprocess(t) for t in df[text_col]]
    labels = df[label_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    results = {}

    # Naive Bayes
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_tr, y_train)
    y_nb = nb.predict(X_te)
    print("\n=== Naive Bayes ===")
    print(classification_report(y_test, y_nb))
    results['naive_bayes'] = {
        'f1_macro': f1_score(y_test, y_nb, average='macro'),
        'report': classification_report(y_test, y_nb, output_dict=True)
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_tr, y_train)
    y_lr = lr.predict(X_te)
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_lr))
    results['logistic_regression'] = {
        'f1_macro': f1_score(y_test, y_lr, average='macro'),
        'report': classification_report(y_test, y_lr, output_dict=True),
        'probas': lr.predict_proba(X_te).tolist()
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {save_path}")

    return nb, lr, vec, y_test