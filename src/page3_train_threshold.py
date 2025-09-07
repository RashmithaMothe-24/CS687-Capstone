"""
Page 3 — Train & Threshold
- Load processed training data.
- Train an anomaly detector (Isolation Forest or LOF).
- Compute anomaly scores.
- Choose an operating threshold (target recall if labels present, else high-quantile).
- Save the trained model and metadata for later pages.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from src.models import (
    fit_isoforest,
    score_isoforest,
    fit_lof,
    score_lof,
    pick_threshold,
    evaluate,
    save_model,
)


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def _train_and_score(
    algo: str,
    X: pd.DataFrame,
    contamination: float,
) -> (np.ndarray, Any, Dict[str, Any]):
    """
    Train the selected algorithm and return scores, fitted model, and metadata.
    """
    if algo == "IsolationForest":
        model = fit_isoforest(X, contamination=contamination)
        scores = score_isoforest(model, X)
        metadata = {
            "algo": "isoforest",
            "contamination": float(contamination),
            "drop_cols": [],  # populated earlier in preprocessing page if needed
        }
    else:
        model = fit_lof(X, contamination=contamination)
        scores = score_lof(model, X)
        metadata = {
            "algo": "lof",
            "contamination": float(contamination),
            "drop_cols": [],
        }
    return scores, model, metadata


def render():
    st.header("3) Train & Threshold")

    train_path = DATA_DIR / "processed_train.csv"
    if not train_path.exists():
        st.error("Processed training set not found. Please run Page 1 (Data & Preprocessing) first.")
        return

    st.markdown(
        """
This page trains an anomaly detection model on the processed training data
and selects an operating threshold. If labels (`Class`) exist, the threshold is
chosen to meet a target recall; otherwise, a high-quantile rule is used.
        """
    )

    # Controls
    algo = st.selectbox("Algorithm", ["IsolationForest", "LOF"], index=0)
    contamination = st.number_input(
        "Assumed contamination (fraction of anomalies)",
        min_value=0.0001,
        max_value=0.2,
        value=0.001,
        step=0.0001,
        format="%.4f",
    )
    target_tpr = st.slider(
        "Target recall (only used if labels exist)",
        min_value=0.50,
        max_value=0.99,
        value=0.90,
        step=0.01,
    )

    if st.button("Train"):
        try:
            df = pd.read_csv(train_path)
            # Keep only numeric columns for the model
            X = df.select_dtypes(include=[np.number]).copy()

            # Preserve y if available for evaluation/threshold
            y = df["Class"].values.astype(int) if "Class" in df.columns else None

            st.write(f"Training matrix shape: {X.shape}")

            with st.spinner("Training model..."):
                scores, model, metadata = _train_and_score(
                    algo="IsolationForest" if algo == "IsolationForest" else "LOF",
                    X=X,
                    contamination=contamination,
                )

                # Threshold selection
                if y is not None:
                    thr = pick_threshold(scores, y_true=y, target_tpr=float(target_tpr))
                else:
                    thr = float(np.quantile(scores, 0.995))

                # Basic evaluation if labels exist
                metrics = evaluate(scores, y_true=y, thr=thr)

            # Display
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Threshold", f"{thr:.6f}")
            with c2:
                if metrics["auc"] is not None:
                    st.metric("AUC", f"{metrics['auc']:.4f}")
            with c3:
                if metrics["precision_at_thr"] is not None:
                    st.metric("Precision@thr", f"{metrics['precision_at_thr']:.4f}")
                    st.metric("Recall@thr", f"{metrics['recall_at_thr']:.4f}")

            st.subheader("Anomaly score distribution (training)")
            st.bar_chart(pd.DataFrame({"anomaly_score": scores}))

            # Save model bundle
            metadata["threshold"] = float(thr)
            model_path = MODELS_DIR / "anomaly_model.joblib"
            save_model(model, str(model_path), metadata)
            st.success(f"Model saved to {model_path}")

            st.info("Proceed to Page 4: Evaluation.")
        except Exception as e:
            st.error(f"Training failed: {e}")
