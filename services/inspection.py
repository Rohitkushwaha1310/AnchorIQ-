import pandas as pd
import numpy as np


def inspect_dataset(df: pd.DataFrame, analysis_type: str = "auto", target_column: str | None = None) -> dict:
    n_rows, n_cols = df.shape

    numerical_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

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

    # Use explicit target if valid, else guess
    if target_column and target_column in df.columns:
        target = target_column
    else:
        target = _guess_target(df, analysis_type, exclude_cols=id_cols)

    stats = {}
    for col in numerical_cols[:12]:
        s = df[col].dropna()
        if len(s) > 0:
            stats[col] = {
                "mean"    : round(float(s.mean()), 2),
                "median"  : round(float(s.median()), 2),
                "std"     : round(float(s.std()), 2) if len(s) > 1 else 0.0,
                "min"     : round(float(s.min()), 2),
                "max"     : round(float(s.max()), 2),
                "skewness": round(float(s.skew()), 2) if len(s) > 2 else 0.0,
                "nulls"   : int(df[col].isnull().sum()),
            }

    # Summary of target distribution if available
    target_info = {}
    if target and target in df.columns:
        t_series = df[target].dropna()
        target_info = {
            "name": target,
            "unique_count": int(t_series.nunique()),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(t_series)),
            "top_values": t_series.value_counts().head(5).to_dict(),
        }

    return {
        "rows"            : n_rows,
        "columns"         : n_cols,
        "numerical_cols"  : numerical_cols,
        "categorical_cols": categorical_cols,
        "id_cols"         : id_cols,
        "missing_total"   : missing_total,
        "missing_cols"    : missing_cols,
        "duplicates"      : duplicates,
        "target_column"   : target,
        "target_info"     : target_info,
        "stats"           : stats,
        "analysis_type"   : analysis_type,
    }


def _guess_target(df: pd.DataFrame, analysis_type: str = "auto", exclude_cols: list | None = None) -> str | None:
    """Guess the target column based on analysis type hint and column names."""
    exclude = set(exclude_cols or [])

    # Type-specific priority columns
    type_targets = {
        "churn"  : ["churn", "churned", "attrition", "cancelled", "left", "churn_binary"],
        "sales"  : ["sales", "revenue", "amount", "price", "total", "quantity", "orders", "profit"],
        "fraud"  : ["fraud", "is_fraud", "fraudulent", "label", "anomaly", "suspicious", "is_fraudulent"],
        "hr"     : ["attrition", "left", "resigned", "promoted", "terminated", "turnover"],
        "general": [],
    }

    # Generic fallback targets
    generic_targets = [
        "target", "label", "class", "outcome", "result", "status",
        "churn", "fraud", "converted", "default", "survived",
        "purchased", "clicked", "is_fraud", "attrition",
    ]

    candidates = [c for c in df.columns if c not in exclude]
    lower_cols = {c.lower(): c for c in candidates}

    # Try type-specific first
    priority = type_targets.get(analysis_type, [])
    for name in priority:
        if name in lower_cols:
            return lower_cols[name]

    # Try generic list
    for name in generic_targets:
        if name in lower_cols:
            return lower_cols[name]

    # Last resort: last column with 2-10 unique values (likely a classification label)
    for col in reversed(candidates):
        n_u = df[col].nunique(dropna=True)
        if 2 <= n_u <= 10:
            return col

    return None


if __name__ == "__main__":
    df = pd.read_csv("telco_churn.csv")
    result = inspect_dataset(df)
    print("=== INSPECTION RESULT ===")
    print(f"Rows           : {result['rows']}")
    print(f"Columns        : {result['columns']}")
    print(f"Target detected: {result['target_column']}")
    print(f"Numerical cols : {result['numerical_cols']}")
    print(f"Missing total  : {result['missing_total']}")
    print(f"Duplicates     : {result['duplicates']}")