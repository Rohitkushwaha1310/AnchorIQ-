"""
Module 2 — Automatic Data Inspection
Replaces manual df.info() / df.describe() digging with one function
that profiles any dataframe and returns a structured JSON-able dict.
"""
import pandas as pd
import numpy as np


def _guess_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Try to detect columns that are dates but stored as strings/objects."""
    candidates = []
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            # if most of the sample parses cleanly, call it a datetime column
            if parsed.notna().mean() > 0.8:
                candidates.append(col)
        except Exception:
            continue
    return candidates


def _guess_target_column(df: pd.DataFrame) -> str | None:
    """
    Heuristic target detector. Looks for common target-like names first,
    falls back to the last low-cardinality categorical/binary column.
    """
    common_names = [
        "target", "label", "class", "churn", "outcome", "y",
        "default", "result", "status", "fraud", "converted",
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for name in common_names:
        for lc, original in lower_cols.items():
            if name in lc:
                return original

    # fallback: last column with 2-10 unique values
    for col in reversed(df.columns):
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 10:
            return col
    return None


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Automatically profile a dataframe. No manual df.info() needed.
    Returns a dict that's directly JSON-serializable for the API response.
    """
    n_rows, n_cols = df.shape

    missing_per_col = df.isna().sum()
    missing_total = int(missing_per_col.sum())

    duplicates = int(df.duplicated().sum())

    datetime_like = _guess_datetime_columns(df)
    numerical_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if c not in datetime_like
    ]

    target_col = _guess_target_column(df)

    result = {
        "rows": n_rows,
        "columns": n_cols,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "missing_total": missing_total,
        "missing_by_column": {
            c: int(v) for c, v in missing_per_col.items() if v > 0
        },
        "duplicates": duplicates,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_like,
        "unique_counts": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        "target_column_guess": target_col,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    return result
