"""
Page 1 — Data & Preprocessing
- Load a single CSV (your credit card transactions dataset).
- Perform robust preprocessing (Amount transform + Time features + numeric sanitization).
- Chronologically split into train/validation to avoid leakage.
- Save processed datasets and preprocessing artefacts.

This module exposes `render()` which is called by the entry app.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Use absolute import so it works when running App.py from project root
from src.pipelines import run_preprocess, save_pipeline


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def _chronological_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split by 'Time' if present; otherwise by existing index order.
    Returns original columns (including labels) because downstream pages may use `Class`.
    """
    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    split = int(train_frac * n)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _infer_default_drop(columns: List[str]) -> List[str]:
    """
    Default columns to drop from model features. Always drop 'Class' if present.
    """
    defaults = []
    if "Class" in columns:
        defaults.append("Class")
    return defaults


def render():
    st.header("1) Data & Preprocessing")

    st.markdown(
        """
This page loads your dataset, applies robust preprocessing, performs a chronological split,
and saves artefacts for reuse by later pages.

**Expected columns**: `Time`, `Amount`, `V1..V28`, optional `Class`.
Outputs are saved to:
- `data/processed_train.csv`
- `data/processed_valid.csv`
- `models/pipeline.joblib`
        """
    )

    # ----------------------------------------------------------------------------------
    # 1) Load data
    # ----------------------------------------------------------------------------------
    st.subheader("Load Data")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    with c2:
        default_path = DATA_DIR / "creditcard.csv"
        use_default = st.checkbox("Use creditcard.csv from data/", value=default_path.exists())

    df = None
    source = None

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            source = "uploaded CSV"
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return
    elif use_default and default_path.exists():
        try:
            df = pd.read_csv(default_path)
            source = str(default_path)
        except Exception as e:
            st.error(f"Failed to read data/creditcard.csv: {e}")
            return

    if df is None:
        st.info("Upload one CSV or tick 'Use creditcard.csv from data/' (place file at data/creditcard.csv).")
        return

    st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {source}.")
    st.dataframe(df.head(20), use_container_width=True)

    # ----------------------------------------------------------------------------------
    # 2) Schema checks
    # ----------------------------------------------------------------------------------
    st.subheader("Schema & Basic Checks")
    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Rows", f"{len(df):,}")
    with c4:
        st.metric("Columns", f"{df.shape[1]}")
    with c5:
        st.metric("Has label 'Class'", "Yes" if "Class" in df.columns else "No")

    st.markdown("Column dtypes")
    st.write(df.dtypes.astype(str))

    st.markdown("Missing values per column")
    st.write(df.isna().sum())

    with st.expander("Descriptive statistics (numeric)"):
        st.write(df.describe().T)

    # ----------------------------------------------------------------------------------
    # 3) Preprocessing configuration
    # ----------------------------------------------------------------------------------
    st.subheader("Preprocessing Configuration")
    default_drop = _infer_default_drop(list(df.columns))
    drop_cols = st.multiselect(
        "Columns to drop from the feature matrix (they remain in saved CSVs for reference):",
        options=list(df.columns),
        default=default_drop,
    )

    if "Amount" in df.columns:
        st.caption("Preview: first 500 entries of Amount")
        st.line_chart(df["Amount"].head(500))

    run_btn = st.button("Run Preprocessing")

    if not run_btn:
        return

    # ----------------------------------------------------------------------------------
    # 4) Execute preprocessing
    # ----------------------------------------------------------------------------------
    try:
        X, artefacts = run_preprocess(df, drop_cols=drop_cols)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    st.success("Preprocessing complete.")
    st.markdown("Transformed feature preview")
    st.dataframe(X.head(20), use_container_width=True)

    if "Amount_scaled" in X.columns:
        st.caption("Preview: Amount_scaled (first 500)")
        st.line_chart(X["Amount_scaled"].head(500))
    if "time_z" in X.columns:
        st.caption("Preview: time_z (first 500)")
        st.line_chart(X["time_z"].head(500))

    # ----------------------------------------------------------------------------------
    # 5) Chronological split and save
    # ----------------------------------------------------------------------------------
    st.subheader("Save Train/Validation and Pipeline")

    train_df, valid_df = _chronological_split(df, train_frac=0.8)

    train_path = DATA_DIR / "processed_train.csv"
    valid_path = DATA_DIR / "processed_valid.csv"
    pipeline_path = MODELS_DIR / "pipeline.joblib"

    try:
        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        save_pipeline(str(pipeline_path), artefacts)
    except Exception as e:
        st.error(f"Saving artefacts failed: {e}")
        return

    st.success(f"Saved:\n- {train_path}\n- {valid_path}\n- {pipeline_path}")
    st.info("Proceed to Page 2: Feature Engineering.")
