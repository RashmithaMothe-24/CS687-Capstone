"""
Page 2 — Feature Engineering
- Load the processed training set.
- Inspect correlations among features.
- Export a manifest of selected features for reproducibility.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")


def render():
    st.header("2) Feature Engineering")

    st.markdown(
        """
This page inspects the processed training set, explores feature correlations,
and exports a manifest of the features that will be used in training.
        """
    )

    train_path = DATA_DIR / "processed_train.csv"
    if not train_path.exists():
        st.error("Processed training set not found. Please run Page 1 (Data & Preprocessing) first.")
        return

    # Load processed training data
    df = pd.read_csv(train_path)
    st.success(f"Loaded processed_train.csv with {df.shape[0]:,} rows × {df.shape[1]} columns.")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Correlation Matrix (Spearman)")
    try:
        corr = df.corr(method="spearman")
        st.dataframe(corr, use_container_width=True)
    except Exception as e:
        st.error(f"Correlation failed: {e}")

    # Drop label if present when exporting features
    feature_cols = [c for c in df.columns if c.lower() != "class"]
    manifest_path = DATA_DIR / "feature_manifest.json"

    if st.button("Save feature manifest"):
        try:
            pd.Series(feature_cols).to_json(manifest_path)
            st.success(f"Saved {manifest_path} with {len(feature_cols)} feature columns.")
        except Exception as e:
            st.error(f"Saving feature manifest failed: {e}")
