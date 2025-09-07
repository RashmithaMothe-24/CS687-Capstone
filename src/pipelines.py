from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from joblib import dump, load


class AmountTransform(BaseEstimator, TransformerMixin):
    """
    Transformer for the 'Amount' column:
    - Apply log1p transformation
    - Scale using RobustScaler (robust to outliers)
    """
    def __init__(self, column: str = "Amount"):
        self.column = column
        self.scaler = RobustScaler()

    def fit(self, X: pd.DataFrame, y=None):
        if self.column in X.columns:
            values = np.log1p(
                X[self.column].astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0).values
            )
            self.scaler.fit(values.reshape(-1, 1))
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.column in X.columns:
            values = np.log1p(
                X[self.column].astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0).values
            )
            X[self.column + "_log1p"] = values
            X[self.column + "_scaled"] = self.scaler.transform(values.reshape(-1, 1)).ravel()
        return X


class TimeFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer for the 'Time' column:
    - Standardize (z-score)
    - Extract cyclical hour-of-day features (sine/cosine encoding)
    """
    def __init__(self, column: str = "Time"):
        self.column = column
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, X: pd.DataFrame, y=None):
        if self.column in X.columns:
            values = X[self.column].astype(float)
            self.mean_ = float(values.mean())
            self.std_ = float(values.std() if values.std() > 0 else 1.0)
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.column in X.columns:
            values = X[self.column].astype(float).values
            time_z = (values - self.mean_) / (self.std_ if self.std_ != 0 else 1.0)
            hours = (values % 86400) / 3600.0  # assuming seconds in 'Time'

            X["time_z"] = time_z
            X["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            X["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        return X


def sanitize_numeric(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """
    Keep only numeric columns, drop specified ones, replace NaN/inf with zero.
    """
    drop_cols = drop_cols or []
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric


def run_preprocess(df: pd.DataFrame, drop_cols=None):
    """
    Run preprocessing pipeline on the dataframe.
    Returns processed features and artefacts for saving.
    """
    drop_cols = drop_cols or []
    X = df.copy()

    amt = AmountTransform()
    tfe = TimeFeatures()

    amt.fit(X)
    X = amt.transform(X)

    tfe.fit(X)
    X = tfe.transform(X)

    features = sanitize_numeric(X, drop_cols=(drop_cols or []) + (["Class"] if "Class" in X.columns else []))

    artefacts = {
        "amount_scaler_center": getattr(amt.scaler, "center_", None),
        "amount_scaler_scale": getattr(amt.scaler, "scale_", None),
        "time_mean": getattr(tfe, "mean_", None),
        "time_std": getattr(tfe, "std_", None),
        "drop_cols": drop_cols or []
    }

    return features, artefacts


def save_pipeline(path: str, artefacts: dict):
    """
    Save preprocessing artefacts.
    """
    dump(artefacts, path)


def load_pipeline(path: str):
    """
    Load preprocessing artefacts.
    """
    return load(path)
