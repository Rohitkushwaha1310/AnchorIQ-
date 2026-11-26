"""
tests/test_api_and_services.py
Comprehensive pytest suite testing:
1. Normal small CSV file
2. File with nulls and duplicates
3. Empty file
4. One-column file
5. Target with a single unique value
6. File with fewer than 30 rows
7. telco_churn.csv end-to-end regression test
8. Upload safeguards: 25 MB rejection (413) and non-allowed extension .exe rejection (400)
"""
import io
import os
import pytest
import pandas as pd
from tests.conftest import df_to_csv_bytes

from services.inspection import inspect_dataset
from services.cleaning import auto_clean
from services.modeling import auto_model
from services.insights import generate_insights
from services.eda import auto_eda


def test_normal_small_csv_services(small_normal_df):
    """Verify services work directly on a clean small dataset."""
    inspection = inspect_dataset(small_normal_df, analysis_type="churn")
    assert inspection["rows"] == len(small_normal_df)
    assert inspection["target_column"] == "Churn"

    cleaned_df, cleaning_rep = auto_clean(small_normal_df.copy())
    assert cleaned_df.shape[0] == small_normal_df.shape[0]

    model_res = auto_model(cleaned_df, target="Churn", analysis_type="churn", session_id="test_normal")
    assert "best_model" in model_res
    assert "error" not in model_res

    insights = generate_insights(inspection, model_res, target="Churn", analysis_type="churn")
    assert "Executive Summary" in insights


def test_normal_small_csv_api(api_client, small_normal_df):
    """Verify POST /analyze works on a normal small CSV via TestClient."""
    csv_bytes = df_to_csv_bytes(small_normal_df)
    resp = api_client.post(
        "/analyze",
        files={"file": ("small_customers.csv", csv_bytes, "text/csv")},
        data={"analysis_type": "churn"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "inspection" in data
    assert "cleaning" in data
    assert "model_results" in data
    assert "insights" in data
    assert data["model_results"].get("best_model") is not None


def test_nulls_and_duplicates_services(nulls_and_duplicates_df):
    """Verify cleaning and pipeline handle duplicates and missing values gracefully."""
    inspection = inspect_dataset(nulls_and_duplicates_df, analysis_type="churn")
    assert inspection["duplicates"] > 0
    assert inspection["missing_total"] > 0

    cleaned_df, cleaning_rep = auto_clean(nulls_and_duplicates_df.copy())
    assert cleaning_rep["nulls_remaining"] == 0
    assert cleaned_df.shape[0] < nulls_and_duplicates_df.shape[0]


def test_nulls_and_duplicates_api(api_client, nulls_and_duplicates_df):
    """Verify POST /analyze handles nulls and duplicates and returns 200 without crashing."""
    csv_bytes = df_to_csv_bytes(nulls_and_duplicates_df)
    resp = api_client.post(
        "/analyze",
        files={"file": ("dirty_data.csv", csv_bytes, "text/csv")},
        data={"analysis_type": "churn"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cleaning"]["nulls_remaining"] == 0
    assert any("duplicate" in step.lower() for step in data["cleaning"]["steps_applied"])


def test_empty_file_api(api_client):
    """Verify empty file upload returns a clear HTTP 400 error and never crashes."""
    resp = api_client.post(
        "/analyze",
        files={"file": ("empty.csv", b"", "text/csv")},
        data={"analysis_type": "auto"}
    )
    assert resp.status_code == 400
    assert "empty" in resp.json().get("detail", "").lower()


def test_one_column_file_api(api_client, one_column_df):
    """Verify one-column dataset returns a sensible response or clear error without crashing."""
    csv_bytes = df_to_csv_bytes(one_column_df)
    resp = api_client.post(
        "/analyze",
        files={"file": ("single_col.csv", csv_bytes, "text/csv")},
        data={"analysis_type": "auto"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["inspection"]["columns"] == 1
    assert "error" in data["model_results"]


def test_single_unique_target_api(api_client, single_unique_target_df):
    """Verify target with only 1 unique class returns a clear message rather than throwing an exception."""
    csv_bytes = df_to_csv_bytes(single_unique_target_df)
    resp = api_client.post(
        "/analyze",
        files={"file": ("constant_target.csv", csv_bytes, "text/csv")},
        data={"analysis_type": "churn", "target_column": "Churn"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data["model_results"]
    assert "unique value" in data["model_results"]["error"].lower()


def test_fewer_than_30_rows_api(api_client, fewer_than_30_rows_df):
    """Verify small file (<30 rows) produces a valid analysis or clean small-sample warning without crashing."""
    csv_bytes = df_to_csv_bytes(fewer_than_30_rows_df)
    resp = api_client.post(
        "/analyze",
        files={"file": ("small_sample.csv", csv_bytes, "text/csv")},
        data={"analysis_type": "churn"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["inspection"]["rows"] == len(fewer_than_30_rows_df)
    assert "model_results" in data


def test_telco_churn_end_to_end(api_client):
    """Verify existing endpoint response shape and full workflow on telco_churn.csv."""
    telco_path = "telco_churn.csv"
    if not os.path.exists(telco_path):
        pytest.skip("telco_churn.csv not found in working directory")

    with open(telco_path, "rb") as f:
        content = f.read()

    resp = api_client.post(
        "/analyze",
        files={"file": ("telco_churn.csv", content, "text/csv")},
        data={"analysis_type": "churn"}
    )
    assert resp.status_code == 200
    data = resp.json()

    expected_keys = [
        "session_id",
        "filename",
        "analysis_type",
        "inspection",
        "cleaning",
        "chart_paths",
        "chart_urls",
        "model_results",
        "insights",
        "status"
    ]
    for key in expected_keys:
        assert key in data, f"Missing key in response: {key}"

    assert data["inspection"]["target_column"] == "Churn"
    assert data["model_results"].get("best_model") in [
        "Logistic Regression", "Random Forest", "XGBoost", "HistGradientBoosting", "LightGBM",
        "Ensemble", "CatBoost", "Extra Trees", "Decision Tree", "Ridge"
    ]
    assert len(data["chart_paths"]) > 0


def test_telco_all_enabled_models_and_all_charts(api_client):
    """Verify that every enabled model is trained without skipping and all charts are generated for Telco."""
    telco_path = "telco_churn.csv"
    if not os.path.exists(telco_path):
        pytest.skip("telco_churn.csv not found in working directory")

    with open(telco_path, "rb") as f:
        content = f.read()

    resp = api_client.post(
        "/analyze",
        files={"file": ("telco_churn.csv", content, "text/csv")},
        data={"analysis_type": "churn"}
    )
    assert resp.status_code == 200
    data = resp.json()

    cv_scores = data.get("model_results", {}).get("cv_scores", {})
    if not cv_scores:
        cv_scores = data.get("model_results", {}).get("model_card", {}).get("cv_scores", {})

    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "HistGradientBoosting",
        "Random Forest",
        "Extra Trees",
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "Ensemble",
    ]
    for model_name in expected_models:
        assert model_name in cv_scores, f"Expected '{model_name}' in cv_scores: {list(cv_scores.keys())}"
        res = cv_scores[model_name]
        assert "skipped" not in res, f"Model '{model_name}' should not be skipped: {res}"
        assert "mean" in res or "error" in res, f"Model '{model_name}' missing evaluation: {res}"

    chart_paths = data.get("chart_paths") or data.get("chart_urls") or []
    assert len(chart_paths) >= 6, f"Expected all 5 EDA charts + SHAP chart, got {len(chart_paths)}"


def test_upload_safeguards_invalid_extension(api_client):
    """Verify disallowed extension (.exe) is rejected with HTTP 400 and a clear error message."""
    resp = api_client.post(
        "/analyze",
        files={"file": ("malicious.exe", b"fake binary executable content", "application/octet-stream")},
        data={"analysis_type": "auto"}
    )
    assert resp.status_code == 400
    assert "unsupported file extension" in resp.json().get("detail", "").lower()


def test_upload_safeguards_file_too_large(api_client, monkeypatch):
    """Verify payload exceeding the max limit is rejected with HTTP 413 and a clear message."""
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "1")
    large_payload = b"x" * (2 * 1024 * 1024)

    resp = api_client.post(
        "/analyze",
        files={"file": ("large_file.csv", large_payload, "text/csv")},
        data={"analysis_type": "auto"}
    )
    assert resp.status_code == 413
    assert "exceeds maximum allowed limit" in resp.json().get("detail", "").lower()


def test_business_gate_acceptance(api_client):
    """Verify telco_churn and Superstore-style data are accepted, while iris and weather datasets are declined (422)."""
    with open("telco_churn.csv", "rb") as f:
        resp_telco = api_client.post("/analyze", files={"file": ("telco_churn.csv", f.read(), "text/csv")})
    assert resp_telco.status_code == 200

    superstore_df = pd.DataFrame({
        "Order_ID": [f"ORD-{i}" for i in range(40)],
        "Customer_Name": [f"Customer {i}" for i in range(40)],
        "Sales": [float(i * 15.5 + 50) for i in range(40)],
        "Profit": [float(i * 3.2 + 10) for i in range(40)],
        "Discount": [0.05 if i % 2 == 0 else 0.15 for i in range(40)],
        "Region": ["East" if i % 2 == 0 else "West" for i in range(40)]
    })
    resp_superstore = api_client.post(
        "/analyze",
        files={"file": ("superstore.csv", df_to_csv_bytes(superstore_df), "text/csv")}
    )
    assert resp_superstore.status_code == 200

    iris_df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0] * 8,
        "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6] * 8,
        "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4] * 8,
        "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2] * 8,
        "species": ["setosa"] * 40
    })
    resp_iris = api_client.post(
        "/analyze",
        files={"file": ("iris.csv", df_to_csv_bytes(iris_df), "text/csv")}
    )
    assert resp_iris.status_code == 422
    assert "couldn't find business-related information" in resp_iris.json()["detail"].lower()

    weather_df = pd.DataFrame({
        "temp_c": [12.5, 15.0, 18.2, 20.1] * 10,
        "humidity": [60, 65, 70, 75] * 10,
        "wind_mph": [5.5, 7.2, 10.1, 12.0] * 10,
        "precipitation": [0.0, 1.2, 0.5, 0.0] * 10
    })
    resp_weather = api_client.post(
        "/analyze",
        files={"file": ("weather.csv", df_to_csv_bytes(weather_df), "text/csv")}
    )
    assert resp_weather.status_code == 422
    assert "couldn't find business-related information" in resp_weather.json()["detail"].lower()


def test_model_upgrade_pipeline_and_noise_verdict():
    """Verify single-row prediction works through saved pipeline and noise target produces weak verdict."""
    from services.predictor import predict_single
    import numpy as np

    rng = np.random.RandomState(42)
    n = 100
    noise_df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.randn(n),
        "noise_target": rng.choice([0, 1], size=n)
    })
    res_noise = auto_model(noise_df, "noise_target", session_id="test_pytest_noise")
    assert "model_card" in res_noise
    assert res_noise["model_card"]["reliability_verdict"] == "weak: barely beats baseline"

    import shutil
    try:
        pred_res = predict_single("test_pytest_noise", {"f1": 0.5, "f2": -0.2, "f3": 1.1})
        assert "prediction" in pred_res
        assert "probability" in pred_res
        assert "risk_level" in pred_res
    finally:
        if os.path.exists("models"):
            shutil.rmtree("models", ignore_errors=True)

