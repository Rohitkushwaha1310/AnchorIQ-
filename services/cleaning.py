import datetime as _dt
import pandas as pd
import numpy as np


def auto_clean(df: pd.DataFrame) -> tuple:
    """Auto-clean any dataset. Returns (cleaned_df, cleaning_report)."""

    report = {
        "original_shape": df.shape,
        "steps_applied" : [],
    }

    # Step 1: Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        report["steps_applied"].append(f"Removed {removed} duplicate rows")

    # Step 2: Fix numeric columns stored as strings (excluding ID columns)
    for col in df.columns:
        c_lower = col.lower()
        is_id = any(c_lower.endswith(k) or c_lower == k for k in ["id", "uuid", "guid", "key"])
        if not is_id and not pd.api.types.is_numeric_dtype(df[col]):
            # Check if majority of non-null values can convert to numeric
            s_nonnull = df[col].dropna()
            # real date/time objects (e.g. from Excel) must NOT be turned into numbers
            if len(s_nonnull) > 0 and s_nonnull.map(
                    lambda v: isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date))).all():
                df[col] = pd.to_datetime(df[col], errors="coerce")
                report["steps_applied"].append(f"Recognised '{col}' as datetime")
                continue
            if len(s_nonnull) > 0:
                converted = pd.to_numeric(s_nonnull, errors="coerce")
                if converted.notna().mean() >= 0.8:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    report["steps_applied"].append(f"Converted '{col}' to numeric")

    # Step 3: Strip whitespace from string columns safely
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            except Exception:
                pass

    # Step 4: Parse date/time columns if applicable
    for col in df.columns:
        c_lower = col.lower()
        if ("date" in c_lower or "time" in c_lower or "timestamp" in c_lower) and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                dt_series = pd.to_datetime(df[col], errors="coerce")
                if dt_series.notna().mean() > 0.5:
                    df[col] = dt_series
                    report["steps_applied"].append(f"Converted '{col}' to datetime")
            except Exception:
                pass

    # Step 5: Impute missing values
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                med = float(df[col].median()) if df[col].notna().any() else 0.0
                df[col] = df[col].fillna(med)
                report["steps_applied"].append(
                    f"Filled {null_count} nulls in '{col}' with median ({med:.2f})"
                )
            else:
                modes = df[col].mode()
                fill_val = modes[0] if len(modes) > 0 else "Unknown"
                df[col] = df[col].fillna(fill_val)
                report["steps_applied"].append(
                    f"Filled {null_count} nulls in '{col}' with mode ('{fill_val}')"
                )

    report["cleaned_shape"]   = df.shape
    report["nulls_remaining"] = int(df.isnull().sum().sum())

    if not report["steps_applied"]:
        report["steps_applied"].append("Data was already clean — no changes needed!")

    return df, report


if __name__ == "__main__":
    df = pd.read_csv("telco_churn.csv")
    df_clean, report = auto_clean(df)
    print("=== CLEANING REPORT ===")
    print(f"Original shape : {report['original_shape']}")
    print(f"Cleaned shape  : {report['cleaned_shape']}")
    print(f"Nulls remaining: {report['nulls_remaining']}")
    print(f"\nSteps applied:")
    for step in report["steps_applied"]:
        print(f"  [OK] {step}")
