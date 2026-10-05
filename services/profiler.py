"""
services/profiler.py
Understands a dataset: semantic column types, target detection (with confidence),
leakage detection and an analysis plan.
"""
import warnings

import numpy as np
import pandas as pd

TARGET_HINTS = {
    "churn":   ["churn", "churned", "attrition", "cancelled", "canceled", "left", "exited", "retained"],
    "sales":   ["sales", "revenue", "amount", "total", "profit", "gmv", "turnover", "units_sold", "quantity"],
    "fraud":   ["fraud", "is_fraud", "fraudulent", "anomaly", "suspicious", "chargeback"],
    "hr":      ["attrition", "left", "resigned", "terminated", "turnover", "promoted"],
    "general": [],
}
GENERIC_TARGETS = [
    "target", "label", "class", "outcome", "result", "converted", "conversion", "default",
    "survived", "purchased", "clicked", "price", "salary", "rating", "score", "status",
]
POSITIVE_WORDS = {"yes", "y", "true", "1", "churn", "churned", "fraud", "fraudulent", "left",
                  "default", "positive", "bad", "cancelled", "canceled", "attrited", "exited"}


def profile(df: pd.DataFrame, analysis_type: str = "auto", target_column: str | None = None):
    """Return (df_with_parsed_dates, profile_dict)."""
    df = df.copy()
    n = len(df)
    col_types = {}

    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        lname = c.lower()

        if pd.api.types.is_datetime64_any_dtype(s):
            col_types[c] = "datetime"
        elif s.dtype == bool or (nun == 2 and not pd.api.types.is_float_dtype(s)):
            col_types[c] = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            id_name = any(lname == k or lname.endswith(("_" + k, " " + k)) or lname.endswith(k)
                          for k in ("id", "uuid", "guid", "index"))
            if id_name and nun / max(n, 1) > 0.9:
                col_types[c] = "id"
            elif pd.api.types.is_integer_dtype(s) and nun <= 10:
                col_types[c] = "categorical"
            else:
                col_types[c] = "numeric"
        else:
            parsed = _try_datetime(s)
            if parsed is not None:
                df[c] = parsed
                col_types[c] = "datetime"
                continue
            avg_len = s.dropna().astype(str).str.len().mean() if nun else 0
            ratio = nun / max(n, 1)
            id_name = any(lname == k or lname.endswith(k) for k in ("id", "uuid", "guid", "key"))
            if (id_name and ratio > 0.5) or ratio > 0.95 and avg_len < 40:
                col_types[c] = "id"
            elif avg_len > 40 and ratio > 0.5:
                col_types[c] = "text"
            elif nun <= 50 or ratio < 0.05:
                col_types[c] = "categorical"
            else:
                col_types[c] = "high_cardinality"

    # ── target detection ──────────────────────────────────────────────────
    target, confidence, reason = None, 0.0, "no clear outcome column"
    if target_column and target_column in df.columns:
        target, confidence, reason = target_column, 1.0, "chosen by user"
    else:
        target, confidence, reason = _detect_target(df, col_types, analysis_type)

    problem_type = None
    positive_label = None
    if target:
        problem_type = _problem_type(df[target], col_types.get(target))
        if problem_type == "classification":
            positive_label = _positive_label(df[target])

    # ── time axis / metric for trend analysis ─────────────────────────────
    dt_cols = [c for c, t in col_types.items() if t == "datetime"]
    time_col = dt_cols[0] if dt_cols else None
    metric = None
    if time_col:
        metric = _pick_metric(df, col_types, target, problem_type)

    plan = ["descriptive"]
    if target:
        plan.append("supervised")
    else:
        plan.append("segmentation")
    plan.append("anomalies")
    if time_col and metric:
        plan.append("forecast")

    info = {
        "column_types": col_types,
        "target": target,
        "target_confidence": round(float(confidence), 2),
        "target_reason": reason,
        "problem_type": problem_type,
        "positive_label": positive_label,
        "time_col": time_col,
        "metric": metric,
        "plan": plan,
        "id_cols": [c for c, t in col_types.items() if t == "id"],
        "numeric_cols": [c for c, t in col_types.items() if t == "numeric"],
        "categorical_cols": [c for c, t in col_types.items() if t in ("categorical", "boolean")],
    }
    return df, info


def find_leakage(df: pd.DataFrame, target: str, col_types: dict, problem_type: str):
    """Return {column: reason} for features that look like they reveal the target."""
    leaks = {}
    y = df[target]
    t_low = target.lower().replace("_", "")
    if problem_type == "classification":
        y_codes = pd.factorize(y.astype(str))[0]
    else:
        y_codes = pd.to_numeric(y, errors="coerce").values

    for c in df.columns:
        if c == target or col_types.get(c) in ("id", "text", "datetime"):
            continue
        c_low = c.lower().replace("_", "")
        if len(t_low) >= 4 and (t_low in c_low or c_low in t_low):
            leaks[c] = f"name overlaps with target '{target}'"
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and col_types.get(c) == "numeric":
            try:
                r = np.corrcoef(s.fillna(s.median()).values.astype(float),
                                np.nan_to_num(y_codes.astype(float)))[0, 1]
                if abs(r) > 0.97:
                    leaks[c] = f"almost perfectly correlated with target (r={r:.2f})"
            except Exception:
                pass
        elif col_types.get(c) in ("categorical", "boolean") and problem_type == "classification":
            g = pd.crosstab(s.astype(str), y.astype(str))
            if len(g) > 1 and len(df) / len(g) >= 20:
                purity = g.max(axis=1).sum() / g.values.sum()
                base = y.value_counts(normalize=True).max()
                if purity > 0.985 and purity - base > 0.1:
                    leaks[c] = f"categories almost fully determine the target (purity {purity:.1%})"
    return leaks


# ── helpers ────────────────────────────────────────────────────────────────
def _try_datetime(s: pd.Series):
    sample = s.dropna().astype(str).head(300)
    if sample.empty or sample.str.len().mean() < 6:
        return None
    if not sample.str.contains(r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", case=False, regex=True).mean() > 0.8:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() < 0.9:
            return None
        return pd.to_datetime(s, errors="coerce")


def _problem_type(s: pd.Series, ctype: str | None) -> str:
    nun = s.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"
    if ctype in ("boolean", "categorical") or nun <= 10:
        return "classification"
    return "regression"


def _positive_label(s: pd.Series):
    vals = s.dropna().unique()
    for v in vals:
        if str(v).strip().lower() in POSITIVE_WORDS:
            return v
    counts = s.value_counts()
    return counts.index[-1]  # minority class


def _detect_target(df, col_types, analysis_type):
    hints = TARGET_HINTS.get(analysis_type, [])
    best, best_score, best_reason = None, 0.0, "no clear outcome column"
    n = len(df)
    for i, c in enumerate(df.columns):
        t = col_types.get(c)
        if t in ("id", "text", "datetime", "high_cardinality"):
            continue
        lname = c.lower().replace(" ", "_")
        score, why = 0.0, []
        if any(lname == h for h in hints):
            score += 6; why.append("name matches the selected analysis type")
        elif any(h in lname for h in hints):
            score += 4; why.append("name contains an expected keyword")
        if any(lname == g for g in GENERIC_TARGETS):
            score += 4; why.append("generic outcome name")
        elif any(g in lname for g in GENERIC_TARGETS):
            score += 2
        if t == "boolean":
            score += 1.5; why.append("binary column")
        if i == len(df.columns) - 1:
            score += 1; why.append("last column")
        if score > best_score:
            best, best_score, best_reason = c, score, "; ".join(why)
    if best_score >= 4:
        return best, min(best_score / 8, 0.95), best_reason
    if best_score >= 2.5:
        return best, 0.4, best_reason + " (low confidence: please confirm)"
    return None, 0.0, "no clear outcome column - running descriptive/unsupervised analysis"


def _pick_metric(df, col_types, target, problem_type):
    if target and problem_type == "regression":
        return target
    keys = ["revenue", "sales", "amount", "total", "profit", "price", "quantity", "units", "count", "value"]
    nums = [c for c, t in col_types.items() if t == "numeric" and c != target]
    for k in keys:
        for c in nums:
            if k in c.lower():
                return c
    return nums[0] if nums else None
