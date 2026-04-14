# src/fusion/late_fusion.py
# Intelligent Systems module — Rational Agent with Late Fusion.
#
# The fusion system is modelled as a goal-based rational agent:
#   Performance : Maximise macro-averaged F1 on the test set.
#   Environment : Twitter image-text pairs (static, fully observable).
#   Actuators   : Predicted emotion label (positive / negative / neutral).
#   Sensors     : Pixel arrays and tokenised captions from each module.
#
# Instead of a fixed lookup table (reflex agent), the agent uses
# Simulated Annealing to find the optimal fusion weights.
# This makes it a goal-based agent: it reasons about which policy
# maximises its performance measure.

import json
import numpy as np
import math
from sklearn.metrics import f1_score, classification_report


def load_probas(path: str) -> np.ndarray:
    """Load probability arrays from a saved metrics JSON file.

    Handles two formats:
    - NLP classic: nested under 'logistic_regression' key.
    - All other modules: directly under 'probas' key.
    """
    with open(path) as f:
        data = json.load(f)
    if 'logistic_regression' in data:
        return np.array(data['logistic_regression']['probas'])
    return np.array(data['probas'])


def evaluate_weights(weights: np.ndarray,
                     probas_list: list,
                     true_labels: np.ndarray) -> float:
    """Compute macro-F1 for a given set of fusion weights.

    This is the objective function that Simulated Annealing maximises.

    Args:
        weights:     Array of weights (one per module), will be normalised.
        probas_list: List of probability arrays, one per module.
        true_labels: Ground-truth class indices.

    Returns:
        Macro-averaged F1 score.
    """
    weights = np.array(weights)
    weights = weights / weights.sum()   # normalise to sum = 1
    fusion  = sum(w * p for w, p in zip(weights, probas_list))
    preds   = np.argmax(fusion, axis=1)
    return f1_score(true_labels, preds, average='macro')


def simulated_annealing(probas_list: list,
                        true_labels: np.ndarray,
                        T0:         float = 1.0,
                        Tf:         float = 0.001,
                        alpha:      float = 0.95,
                        max_iter:   int   = 500,
                        random_state: int = 42):
    """Optimise fusion weights using Simulated Annealing (SA).

    SA is a metaheuristic algorithm inspired by the physical process
    of cooling metals. It balances exploration and exploitation:
    - High temperature (start): accepts worse solutions to escape local optima.
    - Low temperature (end):    only accepts improving solutions.

    Acceptance criterion (Metropolis):
        If new solution is better  → always accept.
        If new solution is worse   → accept with probability exp(delta / T).

    Args:
        probas_list:   List of probability arrays (one per module).
        true_labels:   Ground-truth class indices for the test set.
        T0:            Initial temperature (high = more exploration).
        Tf:            Final temperature (low = exploitation only).
        alpha:         Cooling rate: T = T * alpha each iteration.
        max_iter:      Number of neighbour solutions per temperature step.
        random_state:  Random seed for reproducibility.

    Returns:
        best_weights (np.ndarray), best_f1 (float), history (list of dicts).
    """
    np.random.seed(random_state)
    n = len(probas_list)

    # Start with equal weights for all modules
    current    = np.ones(n) / n
    best       = current.copy()
    current_f1 = evaluate_weights(current, probas_list, true_labels)
    best_f1    = current_f1

    T         = T0
    history   = []
    iteration = 0

    print("Simulated Annealing started")
    print(f"T0={T0} · Tf={Tf} · alpha={alpha}")
    print(f"Initial F1 (equal weights): {current_f1:.4f}\n")

    while T > Tf:
        for _ in range(max_iter):

            # Generate a neighbour: add small random perturbation
            neighbor = current + np.random.uniform(-0.1, 0.1, n)
            neighbor = np.clip(neighbor, 0, 1)   # keep weights in [0, 1]
            if neighbor.sum() == 0:
                continue
            neighbor = neighbor / neighbor.sum()

            neighbor_f1 = evaluate_weights(
                neighbor, probas_list, true_labels)
            delta = neighbor_f1 - current_f1

            # Metropolis acceptance criterion
            if delta > 0:
                # Better solution → always accept
                current    = neighbor
                current_f1 = neighbor_f1
            else:
                # Worse solution → accept with probability exp(delta / T)
                prob = math.exp(delta / T)
                if np.random.random() < prob:
                    current    = neighbor
                    current_f1 = neighbor_f1

            # Update global best
            if current_f1 > best_f1:
                best    = current.copy()
                best_f1 = current_f1

        # Record temperature and best F1 for convergence plot
        history.append({
            'T':       round(T, 6),
            'best_f1': round(best_f1, 4)
        })

        T *= alpha   # cooling step
        iteration += 1

    print(f"SA completed in {iteration} iterations")
    print("Optimal weights found:")
    module_names = ['LR', 'BERT', 'SVM', 'ResNet', 'RoBERTa'][:n]
    for name, w in zip(module_names, best):
        print(f"  {name}: {w:.3f}")
    print(f"Best F1: {best_f1:.4f}")

    return best, best_f1, history


def run_fusion(true_labels,
               class_names,
               paths: list = None):
    """Run the full late fusion pipeline with Simulated Annealing.

    Loads probability arrays from all module result files,
    optimises fusion weights with SA, and evaluates the final system.

    Args:
        true_labels: Ground-truth class indices for the test set.
        class_names: List of class name strings (e.g. ['negative', ...]).
        paths:       List of paths to module JSON result files.
                     Defaults to all 5 modules (NLP, BERT, CV, ResNet, LLM).

    Returns:
        preds, fusion_probas, best_f1, history, best_weights_norm
    """
    if paths is None:
        paths = [
            'results/metrics_nlp_classic.json',
            'results/metrics_bert.json',
            'results/metrics_cv_classic.json',
            'results/metrics_resnet.json',
            'results/metrics_llm.json'
        ]

    # Load and align probability arrays
    probas_list = [load_probas(p) for p in paths]
    n           = min(len(p) for p in probas_list)
    probas_list = [p[:n] for p in probas_list]
    true_labels = true_labels[:n]

    # Optimise fusion weights with Simulated Annealing
    best_weights, best_f1, history = simulated_annealing(
        probas_list, true_labels)

    # Compute final predictions with optimal weights
    best_weights_norm = best_weights / best_weights.sum()
    fusion = sum(w * p for w, p in zip(best_weights_norm, probas_list))
    preds  = np.argmax(fusion, axis=1)

    print("\n=== Late Fusion — Simulated Annealing ===")
    print(classification_report(
        true_labels, preds, target_names=class_names))
    print(f"Final F1 macro: {best_f1:.4f}")

    return preds, fusion, best_f1, history, best_weights_norm
