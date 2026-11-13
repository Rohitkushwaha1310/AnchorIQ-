"""Manual prediction engine with strict schema validation, calibrated probabilities and explanations."""
from __future__ import annotations

import os
import io
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from services.explanations import explain_single_prediction

# Named thresholds for risk labeling
RISK_LOW_THRESHOLD = 0.35
RISK_HIGH_THRESHOLD = 0.70


def validate_input_against_schema(input_data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Validate input data against the saved schema.
    Raises ValueError with an explicit, clear message for:
      - Missing fields
      - Unknown categories
      - Out-of-range numerical values
    """
    cols_schema = schema.get("columns") or schema.get("fields") or {}
    expected_fields = schema.get("column_names") or list(cols_schema.keys())

    # 1. Missing fields check
    missing = [f for f in expected_fields if f not in input_data or input_data[f] is None or input_data[f] == ""]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    # 2. Per-field validation
    for col in expected_fields:
        val = input_data[col]
        meta = cols_schema.get(col, {})
        if not isinstance(meta, dict):
            continue

        kind = meta.get("kind")
        if kind == "categorical" or "allowed_categories" in meta:
            allowed = meta.get("allowed_categories")
            if allowed:
                allowed_str = [str(a) for a in allowed]
                if str(val) not in allowed_str:
                    raise ValueError(
                        f"Unknown category '{val}' for field '{col}'. "
                        f"Allowed categories: {', '.join(allowed_str[:25])}"
                    )

        elif kind == "numeric" or "min" in meta or "max" in meta:
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid numeric value '{val}' for field '{col}'.")

            min_val = meta.get("min")
            max_val = meta.get("max")
            if min_val is not None and num_val < min_val:
                raise ValueError(
                    f"Value {num_val} for field '{col}' is out of range. "
                    f"Minimum allowed value is {min_val}."
                )
            if max_val is not None and num_val > max_val:
                raise ValueError(
                    f"Value {num_val} for field '{col}' is out of range. "
                    f"Maximum allowed value is {max_val}."
                )


def load_model_bundle(session_or_run_id: Any, models_dir: str = "models") -> Dict[str, Any]:
    """Find and load a persisted model bundle and its schema."""
    sid = str(session_or_run_id)
    candidates = [
        os.path.join(models_dir, f"{sid}_model.pkl"),
        os.path.join(models_dir, f"{sid}.pkl"),
        os.path.join(models_dir, "model.pkl"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(f"No trained model found for identifier '{session_or_run_id}'.")

    saved = joblib.load(path)
    if not isinstance(saved, dict):
        saved = {"pipeline": saved, "features": [], "problem_type": "classification"}
    return saved


def load_model_bundle_from_bytes(model_bytes: bytes) -> Dict[str, Any]:
    """Load a serialized model directly from remote storage without persisting it locally."""
    if not model_bytes:
        raise FileNotFoundError("The remote model artifact is empty.")
    saved = joblib.load(io.BytesIO(model_bytes))
    if not isinstance(saved, dict):
        saved = {"pipeline": saved, "features": [], "problem_type": "classification"}
    return saved


def predict_single(
    session_id: Any,
    input_data: Dict[str, Any],
    models_dir: str = "models",
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run prediction on a single manual input row.
    Validates against the model's schema, evaluates pipeline, computes calibrated
    probability, classifies risk level using named constants, and computes reasons.
    """
    bundle = bundle or load_model_bundle(session_id, models_dir=models_dir)
    pipeline = bundle.get("pipeline")
    features = bundle.get("features", [])
    target = bundle.get("target", "Target")
    problem_type = bundle.get("problem_type", "classification")
    threshold = float(bundle.get("threshold") or 0.5)
    classes = bundle.get("classes")
    schema = bundle.get("schema") or {}
    model_card = bundle.get("model_card") or {}
    feature_importance = model_card.get("feature_importance") or bundle.get("feature_importance") or {}

    # Validate against schema if available
    if schema:
        validate_input_against_schema(input_data, schema)

    # Construct single-row DataFrame with expected features
    if features:
        row_dict = {f: input_data.get(f, np.nan) for f in features}
    else:
        row_dict = input_data
    df_input = pd.DataFrame([row_dict])

    prob = 0.5
    risk_label = "Medium"

    if problem_type == "classification":
        try:
            if hasattr(pipeline, "predict_proba"):
                proba_arr = pipeline.predict_proba(df_input)
                if proba_arr.shape[1] == 2:
                    prob = float(proba_arr[0, 1])
                else:
                    prob = float(proba_arr[0].max())
            else:
                prob = 0.5
        except Exception:
            prob = 0.5

        conf_pct = f"{abs(prob if prob >= 0.5 else 1.0 - prob) * 100:.1f}%"

        # Binary decision
        if prob >= threshold:
            pred_decision = classes[1] if classes and len(classes) > 1 else 1
        else:
            pred_decision = classes[0] if classes and len(classes) > 1 else 0

        # Named constant threshold risk labeling
        if prob < RISK_LOW_THRESHOLD:
            risk_label = "Low"
        elif prob < RISK_HIGH_THRESHOLD:
            risk_label = "Medium"
        else:
            risk_label = "High"

        # Recommendation logic
        target_str = str(target).lower()
        if "churn" in target_str or "attrition" in target_str:
            if risk_label == "High":
                recommendation = f"High churn risk ({conf_pct} confidence) — offer retention incentive and proactive review immediately."
            elif risk_label == "Medium":
                recommendation = f"Moderate churn risk ({conf_pct} confidence) — monitor activity and queue for loyalty follow-up."
            else:
                recommendation = f"Low churn risk ({conf_pct} confidence) — healthy customer account with stable usage."
        else:
            recommendation = f"Prediction: {pred_decision} with {conf_pct} confidence. Risk level: {risk_label}."

        final_pred = pred_decision

    else:
        # Regression
        pred_val = float(pipeline.predict(df_input)[0])
        final_pred = round(pred_val, 2)
        prob = 1.0
        conf_pct = "Continuous"
        risk_label = "Medium"
        recommendation = f"Estimated outcome for {target}: {final_pred:,.2f}."

    # Top 3 explanations
    raw_pipe = bundle.get("raw_pipeline") or pipeline
    reasons, shap_status = explain_single_prediction(
        pipeline=raw_pipe,
        input_df=df_input,
        features=features,
        top_feature_importance=feature_importance
    )

    return {
        "run_id": session_id,
        "prediction": final_pred,
        "calibrated_probability": round(prob, 4),
        "probability": round(prob, 4),
        "confidence_pct": conf_pct,
        "risk_label": risk_label,
        "risk_level": risk_label,
        "reasons": reasons[:3],
        "shap_status": shap_status,
        "threshold_used": round(threshold, 3),
        "recommendation": recommendation,
    }
