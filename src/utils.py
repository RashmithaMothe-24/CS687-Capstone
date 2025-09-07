"""
src/utils.py

Utility functions for drift detection and feature sanitization.
Includes:
- make_features: drop columns, keep only numeric, sanitize
- psi: Population Stability Index
- ks_stat: Kolmogorov–Smirnov test
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def make_features(df: pd.DataFrame, drop_cols=None) -> pd.DataFrame:
    """
    Prepare features by dropping specified columns and keeping only numeric columns.
    Replace NaN and infinite values with zeros.
    """
    drop_cols = drop_cols or []
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for continuous features.
    Measures the shift between expected (baseline) and actual distributions.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Define bin edges based on expected distribution
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_perc = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-6, 1.0)
    act_perc = np.clip(act_counts / max(act_counts.sum(), 1), 1e-6, 1.0)

    value = np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))
    return float(value)


def ks_stat(expected: np.ndarray, actual: np.ndarray):
    """
    Kolmogorov-Smirnov statistic and p-value between two distributions.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan, np.nan

    stat, p = ks_2samp(expected, actual)
    return float(stat), float(p)
