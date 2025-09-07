# src/data_utils.py
import pandas as pd
from typing import Tuple, List

REQUIRED_COLS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def read_csv_safe(file) -> pd.DataFrame:
    return pd.read_csv(file)

def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    extras = [c for c in df.columns if c not in REQUIRED_COLS + ["Class"]]
    return True, ("OK" if not extras else f"OK (extra cols ignored in training: {extras})")

def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = REQUIRED_COLS + (["Class"] if "Class" in df.columns else [])
    return df[cols].describe().T

def class_balance(df: pd.DataFrame):
    if "Class" not in df.columns:
        return None
    vc = df["Class"].value_counts().sort_index()
    total = int(vc.sum())
    fraud = int(vc.get(1, 0))
    return {"total": total, "fraud": fraud, "ratio": (fraud / total) if total else 0.0}

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[REQUIRED_COLS].copy()

def get_labels(df: pd.DataFrame):
    return df["Class"].values if "Class" in df.columns else None
