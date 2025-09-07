"""
Page 4 — Evaluation
- Load the saved anomaly model and the processed validation set.
- Score validation data and compute metrics.
- Display confusion matrix-style counts at the chosen threshold (if labels exist).
- Visualize score distribution.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.models import load_model, score_isoforest, score_lof, evaluate


DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def render():
    st.header("4) Evaluation")

    valid_path = DATA_DIR / "processed_valid.csv"
    model_path = MODELS_DIR / "anomaly_model.joblib"

    if not valid_path.exists():
        st.error("Validation set not found. Please complete Page 1 (Data & Preprocessing).")
        return
    if not model_path.exists():
        st.error("Saved model not found. Please complete Page 3 (Train & Threshold).")
        return

    st.markdown(
        """
This page evaluates the saved anomaly detector on the holdout validation set.
If labels (`Class`) exist, we report AUC, Precision@threshold, and Recall@threshold,
and show a confusion-matrix style summary at the chosen operating threshold.
        """
    )

    try:
        df = pd.read_csv(valid_path)
        model, meta = load_model(str(model_path))
        X = df.select_dtypes(include=[np.number])
        algo = meta.get("algo", "isoforest")

        # Compute anomaly scores
        if algo == "isoforest":
            scores = score_isoforest(model, X)
        else:
            scores = score_lof(model, X)

        # Threshold: use stored threshold if available, else 99.5th percentile
        thr = float(meta.get("threshold", float(np.quantile(scores, 0.995))))

        # Metrics if labels exist
        y = df["Class"].values.astype(int) if "Class" in df.columns else None
        metrics = evaluate(scores, y_true=y, thr=thr)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Algorithm", "Isolation Forest" if algo == "isoforest" else "LOF")
            st.metric("Threshold", f"{thr:.6f}")
        if metrics["auc"] is not None:
            with c2:
                st.metric("AUC", f"{metrics['auc']:.4f}")
            with c3:
                st.metric("Precision@thr", f"{metrics['precision_at_thr']:.4f}")
                st.metric("Recall@thr", f"{metrics['recall_at_thr']:.4f}")

        st.subheader("Validation score distribution")
        st.bar_chart(pd.DataFrame({"anomaly_score": scores}))

        # Confusion matrix-like table if labels exist
        if y is not None:
            y_pred = (scores >= thr).astype(int)
            tp, fp, tn, fn = _confusion_counts(y, y_pred)
            st.subheader("Confusion matrix counts (at threshold)")
            cm = pd.DataFrame(
                [[tp, fp], [fn, tn]],
                columns=["Predicted Fraud (1)", "Predicted Legit (0)"],
                index=["Actual Fraud (1)", "Actual Legit (0)"],
            )
            st.table(cm)

        st.info("Proceed to Page 5: Live Monitor & Feedback.")
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
