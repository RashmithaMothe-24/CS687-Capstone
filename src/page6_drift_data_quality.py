"""
Page 6 — Drift & Data Quality
- Compare a baseline dataset vs. a live (or recent) window for distribution drift.
- Compute PSI (Population Stability Index) and KS statistics per selected feature.
- Provide quick guidance on drift severity and basic data-quality checks.
"""

from __future__ import annotations
from pathlib import Path
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.utils import psi, ks_stat


DATA_DIR = Path("data")


def _load_df(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CSV: {path} — {e}")
        return pd.DataFrame()


def _drift_label(psi_value: float) -> Tuple[str, str]:
    """
    Return (level, message) based on PSI thresholds.
    """
    if np.isnan(psi_value):
        return "unknown", "Insufficient data to compute PSI."
    if psi_value >= 0.3:
        return "high", "High drift detected (PSI ≥ 0.3). Consider retraining or revising threshold."
    if psi_value >= 0.2:
        return "moderate", "Moderate drift (0.2 ≤ PSI < 0.3). Monitor closely and investigate."
    return "low", "Low drift (PSI < 0.2)."


def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    x = df.select_dtypes(include=[np.number]).copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x


def render():
    st.header("6) Drift & Data Quality")

    st.markdown(
        """
This page measures distribution drift between a **baseline** dataset and a **live** dataset
(or a recent window of it). Start with the processed files saved on Page 1,
or upload your own CSVs.
        """
    )

    # --------------------------------------------------------------------------
    # Inputs
    # --------------------------------------------------------------------------
    st.subheader("Inputs")

    c1, c2 = st.columns(2)
    with c1:
        baseline_path_str = st.text_input(
            "Baseline CSV path",
            value=str((DATA_DIR / "processed_train.csv").resolve()),
        )
        baseline_upload = st.file_uploader("Or upload a baseline CSV", type=["csv"], key="baseline_upload")
    with c2:
        live_path_str = st.text_input(
            "Live CSV path",
            value=str((DATA_DIR / "processed_valid.csv").resolve()),
        )
        live_upload = st.file_uploader("Or upload a live CSV", type=["csv"], key="live_upload")

    # Load baseline
    if baseline_upload is not None:
        try:
            base_df = pd.read_csv(baseline_upload)
        except Exception as e:
            st.error(f"Failed to read uploaded baseline: {e}")
            return
    else:
        base_df = _load_df(Path(baseline_path_str))
        if base_df.empty:
            st.info("Provide a baseline CSV.")
            return

    # Load live
    if live_upload is not None:
        try:
            live_df = pd.read_csv(live_upload)
        except Exception as e:
            st.error(f"Failed to read uploaded live: {e}")
            return
    else:
        live_df = _load_df(Path(live_path_str))
        if live_df.empty:
            st.info("Provide a live CSV.")
            return

    # --------------------------------------------------------------------------
    # Select feature and window
    # --------------------------------------------------------------------------
    st.subheader("Configuration")

    numeric_cols = sorted(set(base_df.select_dtypes(include=[np.number]).columns) & set(live_df.select_dtypes(include=[np.number]).columns))
    if not numeric_cols:
        st.error("No numeric columns common to both datasets. Cannot compute drift.")
        return

    feature = st.selectbox("Feature to inspect", options=numeric_cols, index=numeric_cols.index("Amount") if "Amount" in numeric_cols else 0)
    window_size = st.number_input("Live window size (rows from the end of live dataset)", min_value=50, max_value=max(10000, len(live_df)), value=min(500, len(live_df)), step=50)

    # --------------------------------------------------------------------------
    # Compute drift
    # --------------------------------------------------------------------------
    try:
        base_vals = base_df[feature].astype(float).values
        live_vals = live_df[feature].astype(float).values[-int(window_size):]

        val_psi = psi(base_vals, live_vals, bins=10)
        ks_s, ks_p = ks_stat(base_vals, live_vals)
        level, guidance = _drift_label(val_psi)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("PSI", f"{val_psi:.4f}")
        with c2:
            st.metric("KS statistic", f"{ks_s:.4f}")
        with c3:
            st.metric("KS p-value", f"{ks_p:.4f}")

        if level == "high":
            st.error(guidance)
        elif level == "moderate":
            st.warning(guidance)
        else:
            st.success(guidance)

        st.subheader("Distribution preview (first 1,000 points)")
        st.line_chart(pd.DataFrame({
            "baseline_sample": pd.Series(base_vals[:1000]),
            "live_window": pd.Series(live_vals[:1000])
        }))
    except Exception as e:
        st.error(f"Drift computation failed: {e}")
        return

    # --------------------------------------------------------------------------
    # Data quality checks
    # --------------------------------------------------------------------------
    st.subheader("Basic Data Quality Checks")

    def _dq_summary(df: pd.DataFrame) -> pd.DataFrame:
        s = {
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "n_missing_cells": int(df.isna().sum().sum()),
            "n_zero_variance_cols": int((df.nunique(dropna=False) <= 1).sum()),
        }
        return pd.DataFrame([s])

    c4, c5 = st.columns(2)
    with c4:
        st.caption("Baseline summary")
        st.table(_dq_summary(base_df))
    with c5:
        st.caption("Live summary")
        st.table(_dq_summary(live_df))

    st.caption("Tip: Also monitor the **anomaly score** distribution drift by recomputing scores on both datasets and applying the same PSI/KS logic.")
