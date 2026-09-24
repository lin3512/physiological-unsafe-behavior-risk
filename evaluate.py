"""Classification metrics and validation-only threshold selection."""
from __future__ import annotations

import numpy as np


def evaluate(y, probabilities, threshold=.3):
    from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                                 roc_auc_score, confusion_matrix)
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "recall_unsafe": float(recall_score(y, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix_true_rows_pred_cols": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def select_recall_threshold(y, probabilities):
    """Choose max recall with specificity >= .70, tie-breaking toward lower threshold."""
    thresholds = np.arange(.10, .501, .01)
    scored = []
    for threshold in thresholds:
        metrics = evaluate(y, probabilities, float(threshold))
        cm = metrics["confusion_matrix_true_rows_pred_cols"]
        specificity = cm[0][0] / max(1, sum(cm[0]))
        if specificity >= .70:
            scored.append((metrics["recall_unsafe"], specificity, -threshold, threshold))
    if not scored:
        return .30
    return float(max(scored)[-1])
