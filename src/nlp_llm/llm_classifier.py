# src/nlp_llm/llm_classifier.py
# EN: Third NLP module — Local LLM classifier using RoBERTa.
#     Uses cardiffnlp/twitter-roberta-base-sentiment-latest,
#     a model specifically trained on Twitter data for sentiment analysis.
#     Runs fully locally via HuggingFace Transformers (no API required).
#
# ES: Tercer módulo NLP — Clasificador LLM local usando RoBERTa.
#     Usa cardiffnlp/twitter-roberta-base-sentiment-latest,
#     un modelo entrenado específicamente en datos de Twitter.
#     Se ejecuta completamente en local mediante HuggingFace Transformers.
#
# Why RoBERTa over classical methods?
# / ¿Por qué RoBERTa sobre métodos clásicos?
#   - Domain-specific: trained on Twitter, same domain as MVSA-Single.
#     / Específico del dominio: entrenado en Twitter, mismo dominio que MVSA-Single.
#   - No fine-tuning needed: achieves F1=0.67 out of the box.
#     / No necesita fine-tuning: alcanza F1=0.67 directamente.
#   - Outperforms Naive Bayes (0.58) and Logistic Regression (0.61).
#     / Supera a Naive Bayes (0.58) y Regresión Logística (0.61).

import json
import numpy as np
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import os


def run_llm(df,
            text_col:  str = 'text',
            label_col: str = 'label',
            save_path: str = 'results/metrics_llm.json'):
    """Classify tweet captions using RoBERTa and save results for fusion.
    / Clasificar captions de tweets con RoBERTa y guardar resultados para la fusión.

    Pipeline / Flujo:
        1. Load RoBERTa pipeline locally.
           / Cargar el pipeline de RoBERTa en local.
        2. Reproduce the same 80/20 train/test split as all other modules.
           / Reproducir el mismo split 80/20 que los otros módulos.
        3. Classify each test text and build a probability vector.
           / Clasificar cada texto del test y construir un vector de probabilidades.
        4. Save F1 score and probabilities to JSON for late fusion.
           / Guardar F1 y probabilidades en JSON para la fusión tardía.

    Args / Argumentos:
        df:        DataFrame with text and label columns.
                   / DataFrame con columnas de texto y etiqueta.
        text_col:  Column name for tweet captions. / Nombre de columna para captions.
        label_col: Column name for emotion labels. / Nombre de columna para etiquetas.
        save_path: Path to save JSON results. / Ruta para guardar resultados JSON.

    Returns / Devuelve:
        all_preds, all_probas, y_test, f1
    """
    os.makedirs('results', exist_ok=True)

    # ── Load model ───────────────────────────────────────────────
    # EN: Load once — this downloads the model on first run (~500MB).
    # ES: Cargar una vez — descarga el modelo en la primera ejecución (~500MB).
    print("Loading RoBERTa for Twitter... / Cargando RoBERTa para Twitter...")
    classifier = pipeline(
        task='text-classification',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
        truncation=True,    # truncate texts longer than max_length
        max_length=128      # Twitter texts are short, 128 tokens is enough
    )
    print("Model ready. / Modelo listo.")

    # ── Prepare data ─────────────────────────────────────────────
    # EN: Same split as all other modules — critical for fair comparison.
    # ES: Mismo split que todos los demás módulos — crítico para comparación justa.
    le     = LabelEncoder()
    labels = le.fit_transform(df[label_col].tolist())
    texts  = df[text_col].tolist()

    _, X_test, _, y_test = train_test_split(
        texts, labels,
        test_size=0.2, random_state=42, stratify=labels)

    print(f"Classifying {len(X_test)} texts... / Clasificando {len(X_test)} textos...")

    # EN: Map RoBERTa output labels to our standard class indices.
    # ES: Mapear etiquetas de salida de RoBERTa a nuestros índices de clase estándar.
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    all_preds  = []
    all_probas = []

    # ── Classify each text ───────────────────────────────────────
    for i, text in enumerate(X_test):
        if i % 100 == 0:
            print(f"  {i}/{len(X_test)}")
        try:
            result     = classifier(str(text)[:512])
            pred_label = result[0]['label'].lower()
            pred_idx   = label_map.get(pred_label, 1)  # default to neutral
            score      = result[0]['score']

            # EN: Build a full 3-class probability vector from the top prediction.
            #     RoBERTa returns only the top label, so we distribute
            #     the remaining probability equally to the other two classes.
            # ES: Construir un vector de probabilidad de 3 clases desde la predicción principal.
            #     RoBERTa solo devuelve la etiqueta principal, así que distribuimos
            #     la probabilidad restante igualmente entre las otras dos clases.
            proba           = np.array([0.05, 0.05, 0.05])
            proba[pred_idx] = score
            remaining       = (1 - score) / 2
            for j in range(3):
                if j != pred_idx:
                    proba[j] = remaining
            proba = proba / proba.sum()   # normalise to sum = 1

        except Exception as e:
            # EN: If classification fails, use uniform distribution.
            # ES: Si la clasificación falla, usar distribución uniforme.
            print(f"Error on text {i}: {e}")
            proba    = np.array([1/3, 1/3, 1/3])
            pred_idx = 1   # default to neutral

        all_probas.append(proba.tolist())
        all_preds.append(pred_idx)

    # ── Evaluate ─────────────────────────────────────────────────
    print("\n=== RoBERTa Twitter LLM ===")
    print(classification_report(
        y_test, all_preds, target_names=le.classes_))
    f1 = f1_score(y_test, all_preds, average='macro')
    print(f"F1 macro: {f1:.4f}")

    # ── Save results for late fusion ─────────────────────────────
    # EN: Probabilities are saved so the fusion module can combine them
    #     with the other four modules using Simulated Annealing weights.
    # ES: Las probabilidades se guardan para que el módulo de fusión pueda
    #     combinarlas con los otros cuatro módulos usando pesos de SA.
    results = {
        'f1_macro': f1,
        'probas':   all_probas,
        'model':    'cardiffnlp/twitter-roberta-base-sentiment-latest'
    }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to / Guardado en {save_path}")

    return all_preds, all_probas, y_test, f1
