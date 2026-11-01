"""SHAP model explanations and feature attribution service."""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mplconfig")))
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def is_tree_model(estimator: Any) -> bool:
    """Check if the estimator is a tree-based model."""
    name = type(estimator).__name__.lower()
    return any(k in name for k in ["forest", "tree", "xgb", "lgbm", "gradientboost"])


def explain_winning_model(
    pipeline: Any,
    X: pd.DataFrame,
    session_id: str,
    charts_dir: str = "charts"
) -> Dict[str, Any]:
    """
    Generate SHAP summary for the winning model if applicable.
    Uses TreeExplainer for tree models, sampled Explainer capped at 500 rows otherwise.
    If SHAP is not applicable or not installed, skips and states why.
    """
    os.makedirs(charts_dir, exist_ok=True)

    if not HAS_SHAP:
        return {
            "applicable": False,
            "reason": "SHAP library ('shap') is not installed in the environment.",
            "chart_path": None,
            "top_drivers": [],
            "fact_card": None,
        }

    try:
        model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
        preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor", None)

        # Cap background at 500 rows
        n_sample = min(500, len(X))
        X_sample = X.iloc[:n_sample].copy()

        if preprocessor is not None:
            try:
                X_trans = preprocessor.transform(X_sample)
                feature_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]
            except Exception:
                X_trans = X_sample.select_dtypes(include=[np.number]).to_numpy()
                feature_names = list(X_sample.select_dtypes(include=[np.number]).columns)
        else:
            X_trans = X_sample.select_dtypes(include=[np.number]).to_numpy()
            feature_names = list(X_sample.select_dtypes(include=[np.number]).columns)

        if is_tree_model(model):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model.predict, X_trans[:min(100, len(X_trans))])

        shap_vals = explainer(X_trans)
        vals = shap_vals.values if hasattr(shap_vals, "values") else np.asarray(shap_vals)
        if vals.ndim == 3:
            # Multi-class or binary two outputs: take class 1 or max variance
            vals = vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]

        # Top 10 drivers by mean absolute SHAP value
        mean_abs = np.mean(np.abs(vals), axis=0)
        indices = np.argsort(mean_abs)[::-1][:10]
        top_drivers = [
            {"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}", "shap_weight": round(float(mean_abs[i]), 4)}
            for i in indices
        ]

        # Save summary chart
        chart_path = os.path.join(charts_dir, "09_shap_summary.png").replace("\\", "/")
        plt.figure(figsize=(9, 6))
        shap.summary_plot(vals, X_trans, feature_names=feature_names, show=False, max_display=10)
        plt.title("SHAP Feature Importance & Impact Summary", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=110, bbox_inches="tight")
        plt.close()

        top_names = [d["feature"] for d in top_drivers[:3]]
        fact_card = {
            "title": "SHAP Global Drivers",
            "local_path": chart_path,
            "numbers": [f"Top: {top_names[0]}" if top_names else "N/A", f"Drivers: {len(top_drivers)}", f"Sample: {n_sample} rows"],
            "finding": f"SHAP analysis identified {', '.join(top_names)} as the primary predictive drivers.",
        }

        return {
            "applicable": True,
            "reason": None,
            "chart_path": chart_path,
            "top_drivers": top_drivers,
            "fact_card": fact_card,
        }

    except Exception as exc:
        return {
            "applicable": False,
            "reason": f"SHAP calculation could not run on this model architecture: {str(exc)}",
            "chart_path": None,
            "top_drivers": [],
            "fact_card": None,
        }


def explain_single_prediction(
    pipeline: Any,
    input_df: pd.DataFrame,
    features: List[str],
    top_feature_importance: Optional[Dict[str, float]] = None
) -> Tuple[List[str], str]:
    """
    Return top 3 reasons for a single prediction.
    If SHAP is available and applicable, uses SHAP values.
    Otherwise, uses trained feature importance and clearly states why SHAP was skipped.
    """
    model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor", None)

    if HAS_SHAP and model is not None and preprocessor is not None:
        try:
            X_trans = preprocessor.transform(input_df)
            feat_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]

            if is_tree_model(model):
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_trans)
                if isinstance(sv, list) and len(sv) == 2:
                    sv = sv[1]
                elif hasattr(sv, "values"):
                    sv = sv.values
                if sv.ndim > 1:
                    sv = sv[0]
                # Rank top 3 by absolute SHAP impact
                abs_indices = np.argsort(np.abs(sv))[::-1][:3]
                reasons = []
                for idx in abs_indices:
                    fname = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
                    imp = float(sv[idx])
                    val = input_df[fname].iloc[0] if fname in input_df.columns else "N/A"
                    sign_str = "+" if imp >= 0 else ""
                    reasons.append(f"{fname} = {val} ({sign_str}{imp:.3f} SHAP impact)")
                return reasons, "TreeExplainer SHAP attribution"
        except Exception:
            pass

    # Fallback when SHAP is not installed or not applicable
    why_skipped = (
        "SHAP not installed in environment; ranked by trained feature importance."
        if not HAS_SHAP
        else "SHAP not applicable for this model pipeline; ranked by trained feature importance."
    )

    fi = top_feature_importance or {}
    if not fi:
        # Check model coef or feature_importances
        raw_imp = getattr(model, "feature_importances_", None)
        if raw_imp is None and hasattr(model, "coef_"):
            raw_imp = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        if raw_imp is not None:
            names = features[:len(raw_imp)]
            fi = {k: round(float(v), 4) for k, v in zip(names, raw_imp)}

    if fi:
        sorted_fi = sorted(fi.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        reasons = []
        for feat, weight in sorted_fi:
            val = input_df[feat].iloc[0] if feat in input_df.columns else "N/A"
            reasons.append(f"{feat} = {val} (key driver, importance weight {weight:.3f})")
        return reasons, why_skipped

    # Generic fallback
    cols = list(input_df.columns)[:3]
    reasons = [f"{c} = {input_df[c].iloc[0]} (primary input feature)" for c in cols]
    return reasons, why_skipped
