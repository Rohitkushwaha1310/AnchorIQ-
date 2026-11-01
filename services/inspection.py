import pandas as pd
import numpy as np


def inspect_dataset(df: pd.DataFrame, analysis_type: str = "auto", target_column: str | None = None) -> dict:
    """
    Auto-detects:
    - rows, columns, dtypes
    - numerical_cols, categorical_cols, datetime_cols
    - missing values per column + total
    - duplicate count
    - ID/identifier columns (exclude from ML)
    - target_column (smart auto-detection with 4 priority levels)
    - problem_type: classification or regression
    - target_info: unique count, top values, is_numeric
    - stats per numerical column: mean, median, std, min, max, skewness, nulls count
    Returns: full inspection dict (JSON-serializable)
    """
    n_rows, n_cols = df.shape

    datetime_cols = []
    numerical_cols = []
    categorical_cols = []

    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            datetime_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            numerical_cols.append(c)
        else:
            categorical_cols.append(c)

    missing_per_col = df.isnull().sum()
    missing_total = int(missing_per_col.sum())
    missing_cols = {str(k): int(v) for k, v in missing_per_col[missing_per_col > 0].items()}
    duplicates = int(df.duplicated().sum())

    # Detect high-cardinality ID / identifier columns
    id_cols = []
    for col in df.columns:
        c_lower = col.lower()
        if any(c_lower.endswith(k) or c_lower == k for k in ["id", "uuid", "guid", "key", "index"]):
            if df[col].nunique(dropna=True) / max(n_rows, 1) > 0.6:
                id_cols.append(col)

    # Target column smart auto-detection:
    # Priority 1: user-provided target_column
    # Priority 2: analysis_type keyword match
    # Priority 3: generic names (target, label, class, outcome, result, status)
    # Priority 4: last column with 2-10 unique values
    target = None
    if target_column and target_column in df.columns:
        target = target_column
    else:
        target = _detect_target(df, analysis_type, exclude_cols=id_cols)

    # Problem type & Target info
    target_info = {}
    problem_type = "classification"
    if target and target in df.columns:
        t_series = df[target].dropna()
        n_uniq = int(t_series.nunique())
        is_num = bool(pd.api.types.is_numeric_dtype(t_series))
        target_info = {
            "name": target,
            "unique_count": n_uniq,
            "is_numeric": is_num,
            "top_values": {str(k): int(v) for k, v in t_series.value_counts().head(5).items()},
        }
        if is_num and n_uniq > 10:
            problem_type = "regression"
        else:
            problem_type = "classification"
    else:
        # Default fallback
        problem_type = "classification"

    # Stats per numerical column
    stats = {}
    for col in numerical_cols:
        s = df[col].dropna()
        if len(s) > 0:
            stats[col] = {
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "skewness": round(float(s.skew()), 2) if len(s) > 2 else 0.0,
                "nulls": int(df[col].isnull().sum()),
            }

    dtypes_dict = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "rows": n_rows,
        "columns": n_cols,
        "dtypes": dtypes_dict,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "missing_total": missing_total,
        "missing_cols": missing_cols,
        "duplicates": duplicates,
        "id_cols": id_cols,
        "target_column": target,
        "problem_type": problem_type,
        "target_info": target_info,
        "stats": stats,
        "analysis_type": analysis_type,
    }


def _detect_target(df: pd.DataFrame, analysis_type: str = "auto", exclude_cols: list | None = None) -> str | None:
    exclude = set(exclude_cols or [])
    candidates = [c for c in df.columns if c not in exclude]
    lower_cols = {c.lower(): c for c in candidates}

    # Priority 2: analysis_type keyword match
    type_keywords = {
        "churn": ["churn", "churned", "attrition", "left"],
        "sales": ["revenue", "sales", "amount", "profit"],
        "fraud": ["fraud", "is_fraud", "anomaly"],
        "hr": ["attrition", "resigned", "terminated", "left"],
    }
    kw_list = type_keywords.get(analysis_type.lower(), [])
    for kw in kw_list:
        if kw in lower_cols:
            return lower_cols[kw]

    # Priority 3: generic names (target, label, class, outcome, result, status)
    generic_names = ["target", "label", "class", "outcome", "result", "status"]
    for g in generic_names:
        if g in lower_cols:
            return lower_cols[g]

    # Check common synonyms if analysis_type was auto
    auto_fallbacks = ["churn", "fraud", "attrition", "sales", "revenue"]
    for af in auto_fallbacks:
        if af in lower_cols:
            return lower_cols[af]

    # Priority 4: last column with 2-10 unique values
    for col in reversed(candidates):
        n_u = df[col].nunique(dropna=True)
        if 2 <= n_u <= 10:
            return col

    # Fallback to last column if available
    if candidates:
        return candidates[-1]

    return None