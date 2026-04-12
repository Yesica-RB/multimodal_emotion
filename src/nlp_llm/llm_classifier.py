# src/nlp_llm/llm_classifier.py
import json
import numpy as np
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import os

def run_llm(df, text_col='text', label_col='label',
            save_path='results/metrics_llm.json'):

    os.makedirs('results', exist_ok=True)

    print("Cargando RoBERTa para Twitter...")
    classifier = pipeline(
        task='text-classification',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
        truncation=True,
        max_length=128
    )
    print("Modelo listo")

    le     = LabelEncoder()
    labels = le.fit_transform(df[label_col].tolist())
    texts  = df[text_col].tolist()

    _, X_test, _, y_test = train_test_split(
        texts, labels,
        test_size=0.2, random_state=42, stratify=labels)

    print(f"Clasificando {len(X_test)} textos...")

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    all_preds  = []
    all_probas = []

    for i, text in enumerate(X_test):
        if i % 100 == 0:
            print(f"  {i}/{len(X_test)}")
        try:
            result     = classifier(str(text)[:512])
            pred_label = result[0]['label'].lower()
            pred_idx   = label_map.get(pred_label, 1)
            score      = result[0]['score']

            # Construir vector de probabilidades
            proba = np.array([0.05, 0.05, 0.05])
            proba[pred_idx] = score
            # Distribuir el resto entre las otras dos clases
            remaining = (1 - score) / 2
            for j in range(3):
                if j != pred_idx:
                    proba[j] = remaining
            proba = proba / proba.sum()

        except Exception as e:
            print(f"Error en texto {i}: {e}")
            proba = np.array([1/3, 1/3, 1/3])
            pred_idx = 1

        all_probas.append(proba.tolist())
        all_preds.append(pred_idx)

    print("\n=== RoBERTa Twitter LLM ===")
    print(classification_report(
        y_test, all_preds,
        target_names=le.classes_))
    f1 = f1_score(y_test, all_preds, average='macro')
    print(f"F1 macro: {f1:.4f}")

    results = {
        'f1_macro': f1,
        'probas':   all_probas,
        'model':    'cardiffnlp/twitter-roberta-base-sentiment-latest'
    }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Guardado en {save_path}")

    return all_preds, all_probas, y_test, f1