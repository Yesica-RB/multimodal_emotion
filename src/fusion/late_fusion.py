# src/fusion/late_fusion.py
import json
import numpy as np
import math
from sklearn.metrics import f1_score, classification_report

def load_probas(path):
    with open(path) as f:
        data = json.load(f)
    if 'logistic_regression' in data:
        return np.array(data['logistic_regression']['probas'])
    return np.array(data['probas'])

def evaluate_weights(weights, probas_list, true_labels):
    weights = np.array(weights)
    weights = weights / weights.sum()
    fusion  = sum(w * p for w, p in zip(weights, probas_list))
    preds   = np.argmax(fusion, axis=1)
    return f1_score(true_labels, preds, average='macro')

def simulated_annealing(probas_list, true_labels,
                        T0=1.0, Tf=0.001, alpha=0.95,
                        max_iter=500, random_state=42):
    np.random.seed(random_state)
    n = len(probas_list)

    # Solución inicial: pesos iguales
    current    = np.ones(n) / n
    best       = current.copy()
    current_f1 = evaluate_weights(current, probas_list, true_labels)
    best_f1    = current_f1

    T       = T0
    history = []
    iteration = 0

    print(f"Simulated Annealing iniciado")
    print(f"T0={T0} · Tf={Tf} · alpha={alpha}")
    print(f"F1 inicial (pesos iguales): {current_f1:.4f}\n")

    while T > Tf:
        for _ in range(max_iter):
            # Generar vecino con perturbación aleatoria
            neighbor = current + np.random.uniform(-0.1, 0.1, n)
            neighbor = np.clip(neighbor, 0, 1)
            if neighbor.sum() == 0:
                continue
            neighbor = neighbor / neighbor.sum()

            neighbor_f1 = evaluate_weights(
                neighbor, probas_list, true_labels)
            delta = neighbor_f1 - current_f1

            # Criterio de aceptación de Metropolis
            if delta > 0:
                current    = neighbor
                current_f1 = neighbor_f1
            else:
                prob = math.exp(delta / T)
                if np.random.random() < prob:
                    current    = neighbor
                    current_f1 = neighbor_f1

            if current_f1 > best_f1:
                best    = current.copy()
                best_f1 = current_f1

        history.append({
            'T':       round(T, 6),
            'best_f1': round(best_f1, 4)
        })
        T *= alpha
        iteration += 1

    print(f"SA completado en {iteration} iteraciones")
    print(f"Pesos óptimos encontrados:")
    labels = ['LR', 'BERT', 'SVM', 'ResNet']
    for l, w in zip(labels, best):
        print(f"  {l}: {w:.3f}")
    print(f"Mejor F1: {best_f1:.4f}")

    return best, best_f1, history

def run_fusion(true_labels, class_names, paths=None):
    if paths is None:
        paths = [
            'results/metrics_nlp_classic.json',
            'results/metrics_bert.json',
            'results/metrics_cv_classic.json',
            'results/metrics_resnet.json'
        ]

    probas_list = [load_probas(p) for p in paths]
    n           = min(len(p) for p in probas_list)
    probas_list = [p[:n] for p in probas_list]
    true_labels = true_labels[:n]

    # Optimizar con Simulated Annealing
    best_weights, best_f1, history = simulated_annealing(
        probas_list, true_labels)

    # Fusión final
    best_weights_norm = best_weights / best_weights.sum()
    fusion = sum(w * p for w, p
                 in zip(best_weights_norm, probas_list))
    preds  = np.argmax(fusion, axis=1)

    print("\n=== Late Fusion — Simulated Annealing ===")
    print(classification_report(
        true_labels, preds, target_names=class_names))
    print(f"F1 macro final: {best_f1:.4f}")

    return preds, fusion, best_f1, history, best_weights_norm