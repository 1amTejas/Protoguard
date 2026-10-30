"""Classification metrics (dependency-free)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def confusion_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    """Confusion matrix with accuracy, precision, recall, F1 and false-positive rate."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def ratio(a: float, b: float) -> float:
        return a / b if b else 0.0

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": ratio(fp, fp + tn),
    }


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Area under the ROC curve (ties count half). ``None`` if only one class is present."""
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and scores[ranked[j + 1]] == scores[ranked[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1  # ranks are 1-based; ties share their average rank
        rank_sum += average_rank * sum(y_true[ranked[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
