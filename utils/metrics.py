"""评价指标计算。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn import metrics
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def calculateScore(y: np.ndarray, pred_y: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y).squeeze()
    pred_y = np.asarray(pred_y).squeeze()

    pred_y_binary = (pred_y > 0.5).astype(int)

    cm = confusion_matrix(y, pred_y_binary)
    if cm.shape == (1, 1):
        if y[0] == 0:
            cm = np.array([[cm[0, 0], 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, cm[0, 0]]])
    elif cm.shape in {(2, 1), (1, 2)}:
        cm = np.array([[0, 0], [0, 0]])

    tn, fp, fn, tp = cm.ravel()

    try:
        roc_area = roc_auc_score(y, pred_y)
    except ValueError:
        roc_area = 0.5

    fpr, tpr, thresholds = roc_curve(y, pred_y)
    pre, rec, _ = precision_recall_curve(y, pred_y)
    pr_auc = metrics.auc(rec, pre)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    bacc = balanced_accuracy_score(y, pred_y_binary)
    kappa = cohen_kappa_score(y, pred_y_binary)

    return {
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sn": sensitivity,
        "sp": specificity,
        "bacc": bacc,
        "acc": accuracy,
        "AUC": roc_area,
        "precision": precision,
        "F1": f1_score,
        "AUC_prec_rec": pr_auc,
        "kappa": kappa,
        "true_labels": y.tolist(),
        "pred_probs": pred_y.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }
