import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Precision-Recall AUC.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(auc(recall, precision))


def basic_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float):
    """
    Compute basic evaluation metrics given scores, labels, and a threshold.
    Returns precision, recall, and ROC-AUC.
    """
    y_pred = (scores >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    roc = roc_auc_score(y_true, scores)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(roc)
    }
