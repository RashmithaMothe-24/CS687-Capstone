"""
Page 5 — Live Monitor & Feedback
- Simulate a streaming scenario from a CSV (validation set or a fresh file).
- Score batches of transactions with the saved anomaly model.
- Flag anomalies using the stored operating threshold.
- Capture analyst feedback (Fraud / Legit) into data/feedback_labels.csv.
- Optional: Retrain the model using accumulated feedback (labels only inform threshold).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.models import load_model, fit_isoforest, fit_lof, save_model

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
FEEDBACK_PATH = DATA_DIR / "feedback_labels.csv"


def _ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def _load_stream_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load a CSV either from an uploaded file or from data/processed_valid.csv as default.
    Sort by Time if present to simulate chronological streaming.
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            default = DATA_DIR / "processed_valid.csv"
            if not default.exists():
                st.error("No stream source selected and data/processed_valid.csv not found. Run Page 1 to create it, or upload a CSV.")
                return None
            df = pd.read_csv(default)

        if "Time" in df.columns:
            df = df.sort_values("Time").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to load stream CSV: {e}")
        return None


def _get_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only numeric columns. Replace inf/nan with zeros.
    """
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def _append_feedback(row: pd.Series, is_fraud: int):
    """
    Append a single feedback record to FEEDBACK_PATH with numeric features and user_label.
    """
    r = row.copy()
    # Attach the label column
    r["user_label"] = int(is_fraud)
    # Only keep numeric + label for compactness
    keep = list(r.select_dtypes(include=[np.number]).index) + ["user_label"]
    r = r[keep]
    r.to_frame().T.to_csv(FEEDBACK_PATH, mode="a", index=False, header=not FEEDBACK_PATH.exists())


def _load_feedback() -> pd.DataFrame:
    if FEEDBACK_PATH.exists():
        try:
            return pd.read_csv(FEEDBACK_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _retrain_with_feedback(
    algo: str,
    contamination: float,
    stream_df: pd.DataFrame,
    metadata: dict,
) -> Tuple[Optional[object], Optional[dict], Optional[str]]:
    """
    Refit the same algorithm using the union of current training data and any feedback rows (features only).
    Threshold is not learned from labels unless user_label exists; labels are used only to choose threshold later (Page 3).
    """
    try:
        train_path = DATA_DIR / "processed_train.csv"
        if not train_path.exists():
            return None, None, "processed_train.csv not found. Run Page 1 first."

        train_df = pd.read_csv(train_path)
        X_train = _get_numeric(train_df)

        fb = _load_feedback()
        if not fb.empty:
            X_fb = _get_numeric(fb.drop(columns=[c for c in fb.columns if c.lower() == "class"], errors="ignore"))
            X_union = pd.concat([X_train, X_fb], axis=0, ignore_index=True)
        else:
            X_union = X_train

        if algo == "isoforest":
            new_model = fit_isoforest(X_union, contamination=contamination)
            new_meta = {
                "algo": "isoforest",
                "contamination": float(contamination),
                "drop_cols": metadata.get("drop_cols", []),
                # preserve threshold; user can re-select on Page 3 using accumulated labels
                "threshold": float(metadata.get("threshold", 0.0)),
            }
        else:
            new_model = fit_lof(X_union, contamination=contamination)
            new_meta = {
                "algo": "lof",
                "contamination": float(contamination),
                "drop_cols": metadata.get("drop_cols", []),
                "threshold": float(metadata.get("threshold", 0.0)),
            }

        save_model(new_model, str(MODELS_DIR / "anomaly_model.joblib"), new_meta)
        return new_model, new_meta, None
    except Exception as e:
        return None, None, f"Retraining failed: {e}"


def render():
    _ensure_dirs()
    st.header("5) Live Monitor & Feedback")

    model_path = MODELS_DIR / "anomaly_model.joblib"
    if not model_path.exists():
        st.error("Saved model not found. Please complete Page 3 (Train & Threshold) first.")
        return

    model, meta = load_model(str(model_path))
    algo = meta.get("algo", "isoforest")
    stored_thr = meta.get("threshold", None)
    contamination = float(meta.get("contamination", 0.001))

    st.markdown(
        """
Simulate a live stream of transactions from a CSV, score each batch, and flag anomalies using the model's threshold.
Provide analyst feedback (Fraud / Legit) and optionally retrain with accumulated feedback.
        """
    )

    # ------------------------------------------------------------------------------
    # Stream source
    # ------------------------------------------------------------------------------
    st.subheader("Stream Source")
    uploaded = st.file_uploader("Upload a CSV to stream (optional). If empty, will use data/processed_valid.csv", type=["csv"])
    df = _load_stream_dataframe(uploaded)
    if df is None:
        return

    # ------------------------------------------------------------------------------
    # Scoring controls
    # ------------------------------------------------------------------------------
    st.subheader("Scoring Controls")

    total_rows = len(df)
    batch_size = st.slider("Events per step", min_value=1, max_value=min(500, total_rows), value=min(50, total_rows), step=1)
    start_idx = st.number_input("Start index", min_value=0, max_value=max(total_rows - 1, 0), value=0, step=1)

    end_idx = min(start_idx + batch_size, total_rows)

    X_all = _get_numeric(df)
    X_batch = X_all.iloc[start_idx:end_idx]
    batch_df = df.iloc[start_idx:end_idx].copy()

    # Score current batch
    try:
        if algo == "isoforest":
            scores = -model.score_samples(X_batch)
        else:
            # LOF novelty decision_function returns higher = more normal; invert
            scores = -model.decision_function(X_batch)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        return

    batch_df["anomaly_score"] = scores
    if stored_thr is None:
        thr = float(np.quantile(scores, 0.995))
        st.info("No stored threshold in metadata; using 99.5th percentile of current batch scores.")
    else:
        thr = float(stored_thr)

    batch_df["flagged"] = (batch_df["anomaly_score"] >= thr).astype(int)

    st.caption(f"Showing rows [{start_idx}:{end_idx}) of {total_rows}. Threshold = {thr:.6f}")
    st.dataframe(batch_df.head(100), use_container_width=True)

    flagged = batch_df[batch_df["flagged"] == 1]
    st.subheader(f"Flagged in this step: {len(flagged)}")
    st.dataframe(flagged, use_container_width=True)

    # ------------------------------------------------------------------------------
    # Feedback capture
    # ------------------------------------------------------------------------------
    st.subheader("Analyst Feedback")

    if total_rows > 0:
        row_abs_index = st.number_input(
            "Row index to label (absolute index in the dataset)", min_value=0, max_value=total_rows - 1, value=start_idx, step=1
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Mark as Fraud (1)"):
                _append_feedback(df.iloc[row_abs_index], is_fraud=1)
                st.success(f"Saved feedback: FRAUD for row {row_abs_index} -> {FEEDBACK_PATH}")
        with c2:
            if st.button("Mark as Legit (0)"):
                _append_feedback(df.iloc[row_abs_index], is_fraud=0)
                st.success(f"Saved feedback: LEGIT for row {row_abs_index} -> {FEEDBACK_PATH}")

    # Show current feedback summary
    fb = _load_feedback()
    st.caption(f"Feedback records: {len(fb)}")
    if not fb.empty:
        st.dataframe(fb.tail(25), use_container_width=True)

    # ------------------------------------------------------------------------------
    # Optional retraining
    # ------------------------------------------------------------------------------
    st.subheader("Retrain With Feedback (optional)")
    st.caption("Refits the model using original training data plus any feedback rows (features only).")
    if st.button("Retrain now"):
        new_model, new_meta, err = _retrain_with_feedback(
            algo=algo,
            contamination=contamination,
            stream_df=df,
            metadata=meta,
        )
        if err:
            st.error(err)
        else:
            st.success("Model retrained and saved to models/anomaly_model.joblib. For threshold re-selection, go to Page 3.")
