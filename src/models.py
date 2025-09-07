"""
src/models.py

Anomaly-detection models and utilities for the fraud detection system.
Provides:
- Isolation Forest / LOF training and scoring
- Threshold selection (supervised or unsupervised)
- Basic evaluation at a chosen operating threshold
- Save/load model bundle with metadata
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def fit_isoforest(
    X: pd.DataFrame,
    n_estimators: int = 300,
    contamination: float = 0.001,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train an Isolation Forest model for anomaly detection.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric feature matrix.
    n_estimators : int
        Number of trees.
    contamination : float
        Expected proportion of anomalies in data.
    random_state : int
        Random seed.

    Returns
    -------
    IsolationForest
        Fitted Isolation Forest model.
    """
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X)
    return clf


def fit_lof(
    X: pd.DataFrame,
    n_neighbors: int = 35,
    contamination: float = 0.001,
) -> LocalOutlierFactor:
    """
    Train a Local Outlier Factor (LOF) model in novelty mode.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric feature matrix.
    n_neighbors : int
        Number of neighbors for local density estimation.
    contamination : float
        Expected proportion of anomalies in data.

    Returns
    -------
    LocalOutlierFactor
        Fitted LOF model with novelty=True.
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,  # enables .predict / .decision_function on new data
    )
    lof.fit(X)
    return lof


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score_isoforest(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """
    Compute anomaly scores for Isolation Forest.
    Higher values = more anomalous.

    IsolationForest.score_samples returns higher values for more NORMAL points,
    so we negate it to obtain an anomaly score.
    """
    scores = -model.score_samples(X)
    return scores


def score_lof(model: LocalOutlierFactor, X: pd.DataFrame) -> np.ndarray:
    """
    Compute anomaly scores for LOF.
    LOF.decision_function returns higher values for more NORMAL points,
    so we negate it to obtain an anomaly score.
    """
    try:
        scores = -model.decision_function(X)
    except Exception:
        # Fallback for older sklearn internals
        scores = -model._decision_function(X)
    return scores


# --------------------------------------------------------------------------------------
# Threshold selection and evaluation
# --------------------------------------------------------------------------------------
def pick_threshold(
    scores: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    target_tpr: float = 0.90,
) -> float:
    """
    Select an operating threshold on anomaly scores.

    If labels are available:
        Choose the threshold on scores to achieve approximately target TPR (recall).
        Implementation: set threshold at the (1 - target_tpr) quantile of positive scores.

    If labels are not available:
        Use a high-quantile rule (default: 99.5th percentile).

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores (higher = more anomalous).
    y_true : Optional[np.ndarray]
        Binary labels where 1 indicates fraud. If None, unsupervised selection is used.
    target_tpr : float
        Desired recall for the positive class when labels exist.

    Returns
    -------
    float
        Selected threshold.
    """
    if y_true is None:
        return float(np.quantile(scores, 0.995))

    y_true = np.asarray(y_true).astype(int)
    pos = scores[y_true == 1]
    if pos.size == 0:
        return float(np.quantile(scores, 0.995))

    thr = np.quantile(pos, 1.0 - target_tpr)
    return float(thr)


def evaluate(
    scores: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    thr: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Evaluate anomaly scores at a given threshold. If labels are provided, returns:
    - AUC
    - Precision at threshold
    - Recall at threshold

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores.
    y_true : Optional[np.ndarray]
        Binary labels (1 = fraud). If None, metrics are None.
    thr : Optional[float]
        Operating threshold. If None and y_true is present, uses 99.5th percentile of scores.

    Returns
    -------
    Dict[str, Optional[float]]
        {"auc": float|None, "precision_at_thr": float|None, "recall_at_thr": float|None}
    """
    if y_true is None:
        return {"auc": None, "precision_at_thr": None, "recall_at_thr": None}

    y_true = np.asarray(y_true).astype(int)
    auc = float(roc_auc_score(y_true, scores))

    if thr is None:
        thr = float(np.quantile(scores, 0.995))

    y_pred = (scores >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "auc": auc,
        "precision_at_thr": float(precision),
        "recall_at_thr": float(recall),
    }


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------
def save_model(model_obj: Any, path: str, metadata: Dict[str, Any]) -> None:
    """
    Save a model bundle with metadata to a joblib file.

    Bundle format: {"model": <sklearn model>, "metadata": {...}}
    """
    dump({"model": model_obj, "metadata": metadata}, path)


def load_model(path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a model bundle saved with save_model().

    Returns
    -------
    (model, metadata)
    """
    bundle = load(path)
    return bundle["model"], bundle.get("metadata", {})
