# src/model_utils.py
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import os, json, time
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

MODELS_DIR = "models"
INDEX_PATH = os.path.join(MODELS_DIR, "models_index.json")
PRODUCTION_PATH = os.path.join(MODELS_DIR, "production_model.joblib")

@dataclass
class TrainResult:
    name: str
    model_path: str
    metrics: Dict[str, float]
    threshold: float

def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

def _build_candidate_models() -> Dict[str, object]:
    """Return a dict of 6 candidate estimators (names → sklearn/xgb estimators)."""
    return {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "xgb": XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss"
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2,
            n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.06, max_depth=3
        ),
        "svc": SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=25, weights="distance")
    }

def _build_pipeline(estimator, use_smote: bool) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(sampling_strategy=0.2, random_state=42)))
    steps.append(("model", estimator))
    return Pipeline(steps)

def _choose_threshold(y_true: np.ndarray, y_score: np.ndarray, target: str = "f1") -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1s = (2 * precision * recall) / (precision + recall + 1e-12)
    if target == "f1":
        best_idx = int(np.nanargmax(f1s))
        if best_idx >= len(thresholds):
            best_idx = max(0, best_idx - 1)
        return float(thresholds[best_idx])
    # fallback
    best_idx = int(np.nanargmax(f1s))
    if best_idx >= len(thresholds):
        best_idx = max(0, best_idx - 1)
    return float(thresholds[best_idx])

def _recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float = 0.90) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # find max recall where precision >= min_precision
    mask = precision[:-1] >= min_precision  # thresholds aligns with precision[:-1]
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[:-1][mask]))

def _evaluate(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_score >= thr).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "F1_at_thr": float(f1_score(y_true, y_pred)),
        "Precision_at_thr": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_at_thr": float(recall_score(y_true, y_pred, zero_division=0)),
        "Recall_at_P90": _recall_at_precision(y_true, y_score, 0.90)
    }

def _save_index(index: List[Dict]):
    _ensure_dirs()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

def _load_index() -> List[Dict]:
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def train_single(df: pd.DataFrame, est_name: str, use_smote: bool = True) -> TrainResult:
    from src.data_utils import select_features, get_labels
    _ensure_dirs()
    X = select_features(df).values
    y = get_labels(df)
    if y is None:
        raise ValueError("Training requires 'Class' column (labels).")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    estimators = _build_candidate_models()
    if est_name not in estimators:
        raise ValueError(f"Unknown model '{est_name}'. Options: {list(estimators.keys())}")

    pipe = _build_pipeline(estimators[est_name], use_smote)
    pipe.fit(X_train, y_train)

    val_scores = pipe.predict_proba(X_val)[:, 1]
    thr = _choose_threshold(y_val, val_scores, target="f1")
    metrics = _evaluate(y_val, val_scores, thr)

    ts = int(time.time())
    model_path = os.path.join(MODELS_DIR, f"{est_name}_{ts}.joblib")
    joblib.dump({"pipeline": pipe, "threshold": thr, "name": est_name, "timestamp": ts}, model_path)

    # update index
    index = _load_index()
    index.append({"name": est_name, "path": model_path, "metrics": metrics, "threshold": thr, "timestamp": ts})
    _save_index(index)

    return TrainResult(name=est_name, model_path=model_path, metrics=metrics, threshold=thr)

def train_and_compare(df: pd.DataFrame, use_smote: bool = True, sort_by: str = "PR_AUC") -> Tuple[List[TrainResult], TrainResult]:
    """Train 5–6 models, return all results and the best result by 'sort_by' metric."""
    results: List[TrainResult] = []
    for name in _build_candidate_models().keys():
        try:
            res = train_single(df, name, use_smote=use_smote)
            results.append(res)
        except Exception as e:
            # keep going; record as failed
            results.append(TrainResult(name=name, model_path="", metrics={"error": str(e)}, threshold=0.5))

    # pick best ignoring failures
    valid = [r for r in results if "error" not in r.metrics]
    if not valid:
        raise RuntimeError("All model trainings failed.")
    best = sorted(valid, key=lambda r: r.metrics.get(sort_by, -1.0), reverse=True)[0]

    # set best as production
    promote_to_production(best.model_path)
    return results, best

def promote_to_production(model_path: str):
    _ensure_dirs()
    blob = joblib.load(model_path)
    joblib.dump(blob, PRODUCTION_PATH)

def load_model(path: str = PRODUCTION_PATH):
    blob = joblib.load(path)
    return blob["pipeline"], float(blob["threshold"])

def predict(df: pd.DataFrame, model_path: Optional[str] = None) -> pd.DataFrame:
    from src.data_utils import select_features
    if model_path is None:
        model_path = PRODUCTION_PATH
    pipe, thr = load_model(model_path)
    X = select_features(df).values
    proba = pipe.predict_proba(X)[:, 1]
    label = (proba >= thr).astype(int)
    out = df.copy()
    out["pred_proba"] = proba
    out["pred_label"] = label
    out["flag"] = out["pred_label"]
    return out

def models_index() -> List[Dict]:
    return _load_index()
