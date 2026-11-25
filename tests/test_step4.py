"""
tests/test_step4.py
Step 4 Tests:
1. A/B test on synthetic data:
   - Known difference is significant
   - No difference is not significant
   - Superstore discount groups comparison is sensible
2. Invalid prediction inputs are rejected:
   - Missing fields
   - Unknown categories
   - Out-of-range numerical values
3. Prediction works for Telco and one other business dataset:
   - Returns prediction, calibrated probability, risk label, and top 3 reasons
"""
import os
import pytest
import numpy as np
import pandas as pd
from fastapi import HTTPException

from services.ab_testing import run_ab_test
from services.modeling import auto_model
from app.main import get_predict_schema, predict_with_schema, ab_test, ABTestPayload


def test_ab_test_synthetic_known_difference_and_no_difference():
    rng = np.random.RandomState(42)

    n = 100
    df_diff = pd.DataFrame({
        "group": ["Control"] * n + ["Treatment"] * n,
        "revenue": list(rng.normal(50.0, 5.0, n)) + list(rng.normal(75.0, 5.0, n))
    })
    res_diff = run_ab_test(df=df_diff, group_column="group", metric_column="revenue")

    assert res_diff["significant"] is True, "Known difference must be statistically significant"
    assert res_diff["p_value"] < 0.05
    assert res_diff["metric_type"] == "continuous"
    assert res_diff["cohens_d"] is not None and res_diff["cohens_d"] > 0
    assert res_diff["confidence_interval"] is not None
    assert len(res_diff["confidence_interval"]) == 2
    assert res_diff["group_sizes"]["Control"] == n
    assert res_diff["group_sizes"]["Treatment"] == n
    assert "statistically significant" in res_diff["plain_english_verdict"].lower()

    df_nodiff = pd.DataFrame({
        "group": ["Control"] * n + ["Treatment"] * n,
        "revenue": list(rng.normal(50.0, 5.0, n)) + list(rng.normal(50.0, 5.0, n))
    })
    res_nodiff = run_ab_test(df=df_nodiff, group_column="group", metric_column="revenue")

    assert res_nodiff["significant"] is False, "Identical distributions must not be statistically significant"
    assert res_nodiff["p_value"] >= 0.05
    assert "no statistically significant" in res_nodiff["plain_english_verdict"].lower()

    df_binary = pd.DataFrame({
        "group": ["A"] * 100 + ["B"] * 100,
        "converted": [0] * 90 + [1] * 10 + [0] * 65 + [1] * 35
    })
    res_binary = run_ab_test(df=df_binary, group_column="group", metric_column="converted")
    assert res_binary["significant"] is True
    assert res_binary["metric_type"] == "binary"
    assert res_binary["lift_pct"] > 0

    n_store = 120
    superstore_ab = pd.DataFrame({
        "Discount_Group": ["Low Discount (0-10%)"] * n_store + ["High Discount (20%+)"] * n_store,
        "Profit": list(rng.normal(65.0, 15.0, n_store)) + list(rng.normal(15.0, 15.0, n_store))
    })
    res_superstore = run_ab_test(df=superstore_ab, group_column="Discount_Group", metric_column="Profit")
    assert res_superstore["significant"] is True
    assert abs(res_superstore["cohens_d"]) > 1.0
    assert "statistically significant" in res_superstore["plain_english_verdict"].lower()
    assert len(res_superstore["notes"]) >= 1
    assert "observational data does not prove causation" in res_superstore["notes"][0].lower()


def test_invalid_prediction_inputs_rejected(tmp_path):
    df_train = pd.DataFrame({
        "tenure": [6, 12, 24, 36, 48, 60, 72] * 6,
        "MonthlyCharges": [25.0, 45.0, 65.0, 85.0, 105.0, 50.0, 70.0] * 6,
        "Contract": ["Month-to-month", "One year", "Two year"] * 14,
        "Churn": ["Yes", "No", "No", "No", "Yes", "No", "No"] * 6
    })
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    session_id = "test_val_run"
    auto_model(df_train, target="Churn", save_dir=models_dir, session_id=session_id)

    schema = get_predict_schema(session_id)
    assert "columns" in schema

    with pytest.raises(HTTPException) as exc_missing:
        predict_with_schema(session_id, {"tenure": 12, "MonthlyCharges": 50.0})
    assert exc_missing.value.status_code == 400
    assert "missing" in exc_missing.value.detail.lower()

    with pytest.raises(HTTPException) as exc_cat:
        predict_with_schema(session_id, {
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "Contract": "Centennial Plan"
        })
    assert exc_cat.value.status_code == 400
    assert "unknown category" in exc_cat.value.detail.lower()

    with pytest.raises(HTTPException) as exc_max:
        predict_with_schema(session_id, {
            "tenure": 500,
            "MonthlyCharges": 50.0,
            "Contract": "Month-to-month"
        })
    assert exc_max.value.status_code == 400
    assert "out of range" in exc_max.value.detail.lower()

    with pytest.raises(HTTPException) as exc_min:
        predict_with_schema(session_id, {
            "tenure": 12,
            "MonthlyCharges": -100.0,
            "Contract": "Month-to-month"
        })
    assert exc_min.value.status_code == 400
    assert "out of range" in exc_min.value.detail.lower()


def test_prediction_works_for_telco_and_other_dataset():
    telco_path = "telco_churn.csv"
    assert os.path.exists(telco_path), "telco_churn.csv must exist in workspace"

    df_telco = pd.read_csv(telco_path).dropna().head(150)
    telco_session = "test_telco_run"
    res_telco = auto_model(df_telco, target="Churn", save_dir="models", session_id=telco_session)
    assert "best_model" in res_telco

    telco_schema = get_predict_schema(telco_session)
    assert telco_schema is not None

    feature_cols = [c for c in df_telco.columns if c != "Churn" and c in (telco_schema.get("columns") or telco_schema)]
    valid_telco_input = {col: df_telco.iloc[0][col] for col in feature_cols}

    pred_telco = predict_with_schema(telco_session, valid_telco_input)
    assert "prediction" in pred_telco
    assert "calibrated_probability" in pred_telco
    assert 0.0 <= pred_telco["calibrated_probability"] <= 1.0
    assert pred_telco["risk_label"] in ["Low", "Medium", "High"]
    assert "reasons" in pred_telco
    assert 1 <= len(pred_telco["reasons"]) <= 3

    rng = np.random.RandomState(123)
    n_store = 150
    df_superstore = pd.DataFrame({
        "Sales": rng.uniform(20.0, 500.0, n_store).round(2),
        "Quantity": rng.randint(1, 10, n_store),
        "Discount": rng.choice([0.0, 0.1, 0.2, 0.3], n_store),
        "Segment": rng.choice(["Consumer", "Corporate", "Home Office"], n_store),
        "Region": rng.choice(["East", "West", "Central", "South"], n_store),
        "HighProfit": rng.choice(["Yes", "No"], n_store, p=[0.45, 0.55]),
    })

    superstore_session = "test_superstore_run"
    res_store = auto_model(df_superstore, target="HighProfit", save_dir="models", session_id=superstore_session)
    assert "best_model" in res_store

    store_schema = get_predict_schema(superstore_session)
    assert store_schema is not None

    feature_cols_store = [c for c in df_superstore.columns if c != "HighProfit" and c in (store_schema.get("columns") or store_schema)]
    valid_store_input = {col: df_superstore.iloc[0][col] for col in feature_cols_store}

    pred_store = predict_with_schema(superstore_session, valid_store_input)
    assert "prediction" in pred_store
    assert "calibrated_probability" in pred_store
    assert 0.0 <= pred_store["calibrated_probability"] <= 1.0
    assert pred_store["risk_label"] in ["Low", "Medium", "High"]
    assert "reasons" in pred_store
    assert 1 <= len(pred_store["reasons"]) <= 3

    import shutil
    if os.path.exists("models"):
        shutil.rmtree("models", ignore_errors=True)

