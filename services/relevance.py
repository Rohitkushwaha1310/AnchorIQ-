"""
services/relevance.py
Business relevance gate for AnchorIQ.
Analyzes column names and dataset structure to determine whether an uploaded
dataset is business-related (sales, churn, marketing, hr, finance, etc.)
or non-business (e.g. iris, weather, biology).
"""
import os
import re
import pandas as pd
from typing import Dict, Any, List

BUSINESS_KEYWORDS = {
    "customer", "cust", "client", "sales", "revenue", "price", "order",
    "product", "region", "discount", "profit", "churn", "subscription",
    "employee", "salary", "invoice", "cost", "quantity", "category",
    "payment", "contract", "lead", "deal", "transaction", "account",
    "billing", "user", "vendor", "spend", "margin", "tax", "fee",
    "tenure", "quota", "conversion", "retention", "attrition", "purchase"
}


def _check_groq_relevance(df: pd.DataFrame, columns: List[str]) -> bool | None:
    """
    For borderline cases: single fast LLM call (Groq) to classify as business or not.
    Uses column names and 5 sample rows. Returns True (business), False (not), or None on error.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        sample_str = df.head(5).to_string(index=False)
        prompt = (
            f"Columns: {', '.join(columns)}\n\n"
            f"Sample data:\n{sample_str}\n\n"
            "Question: Is this a business, commercial, financial, HR, sales, or customer dataset? "
            "Reply with ONLY one word: YES or NO."
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=5,
            messages=[
                {"role": "system", "content": "You are a business data classifier. Answer ONLY 'YES' or 'NO'."},
                {"role": "user", "content": prompt}
            ]
        )
        ans = (response.choices[0].message.content or "").strip().upper()
        if "YES" in ans:
            return True
        elif "NO" in ans:
            return False
    except Exception as e:
        print(f"[INFO] Groq relevance check skipped/failed: {e}")

    return None


def check_business_relevance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Determines if dataset is business-related.
    Returns:
        {
            "score": int (0-100),
            "verdict": "business" | "borderline" | "not_business",
            "reasons": list[str],
            "columns_found": list[str]
        }
    """
    if df.empty or len(df.columns) == 0:
        return {
            "score": 0,
            "verdict": "not_business",
            "reasons": ["Dataset is empty."],
            "columns_found": []
        }

    score = 0
    reasons = []
    columns_found = []

    cols = list(df.columns)
    clean_cols = [str(c).lower().strip() for c in cols]

    matched_cols = []
    for orig_col, c_clean in zip(cols, clean_cols):
        tokens = re.split(r"[_\s\-\.]+", c_clean)
        if any(kw in tokens or kw in c_clean for kw in BUSINESS_KEYWORDS):
            matched_cols.append(orig_col)

    if matched_cols:
        columns_found = matched_cols
        pts = min(45, len(matched_cols) * 15)
        score += pts
        reasons.append(f"Found {len(matched_cols)} business keyword column(s): {', '.join(matched_cols[:5])}")

    generic_col_pattern = re.compile(r"^(col|column|var|x|v|feat|feature)[_\-\d]*$", re.IGNORECASE)
    all_generic = all(bool(generic_col_pattern.match(c)) for c in clean_cols)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    has_money_signal = False
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 0 and (s >= 0).all():
            mean_val = float(s.mean())
            max_val = float(s.max())
            if 5.0 <= mean_val <= 500000.0 and max_val > 10.0:
                has_money_signal = True
                break

    if has_money_signal:
        score += 20
        reasons.append("Contains numeric measurements consistent with monetary/transactional metrics.")

    has_date = False
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            has_date = True
            break
        c_lower = str(col).lower()
        if any(d in c_lower for d in ["date", "time", "year", "month", "day", "timestamp"]):
            has_date = True
            break

    if has_date and len(numeric_cols) >= 1:
        score += 20
        reasons.append("Contains temporal date/time dimensions paired with numeric metrics.")

    for col in df.columns:
        c_lower = str(col).lower()
        if any(t in c_lower for t in ["target", "churn", "status", "converted", "default", "fraud", "outcome", "label"]):
            score += 15
            reasons.append(f"Identified plausible outcome target '{col}'.")
            break

    if all_generic:
        score = max(score, 40)
        reasons.append("Column headers appear generic/obfuscated; evaluated via value signals.")

    score = min(100, score)

    if score >= 50:
        verdict = "business"
    elif score >= 25:
        verdict = "borderline"
    else:
        verdict = "not_business"

    if verdict == "borderline":
        groq_is_business = _check_groq_relevance(df, cols)
        if groq_is_business is True:
            verdict = "business"
            score = max(score, 60)
            reasons.append("AI validation confirmed business domain context.")
        elif groq_is_business is False and not all_generic:
            verdict = "not_business"
            score = min(score, 20)
            reasons.append("AI validation evaluated dataset as non-business.")
        elif groq_is_business is False:
            reasons.append("Generic column names prevent a confident rejection; treating the dataset as borderline.")

    return {
        "score": score,
        "verdict": verdict,
        "reasons": reasons,
        "columns_found": columns_found,
    }
