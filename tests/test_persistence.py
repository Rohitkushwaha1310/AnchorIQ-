import importlib
import os

import pandas as pd
from fastapi.testclient import TestClient

from tests.conftest import df_to_csv_bytes


def _reload_app(monkeypatch, database_url="sqlite:///./test_persistence.sqlite", cloudinary_enabled=True):
    monkeypatch.setenv("DATABASE_URL", database_url)
    if cloudinary_enabled:
        monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "demo")
        monkeypatch.setenv("CLOUDINARY_API_KEY", "demo-key")
        monkeypatch.setenv("CLOUDINARY_API_SECRET", "demo-secret")
    else:
        for key in ["CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"]:
            monkeypatch.delenv(key, raising=False)

    import database.db as db_module
    import app.main as app_module

    importlib.reload(db_module)
    importlib.reload(app_module)
    db_module.initialize_database()
    return app_module, db_module


def test_run_is_saved_with_charts_and_insights(monkeypatch):
    app_module, db_module = _reload_app(monkeypatch, cloudinary_enabled=True)

    def fake_upload(file_path, run_id, title):
        return {"secure_url": f"https://example.com/{run_id}/{title}.png", "public_id": f"public_{run_id}_{title}", "uploaded": True}

    monkeypatch.setattr(app_module, "upload_chart_png", fake_upload)

    df = pd.DataFrame(
        {
            "tenure": [12, 18, 24, 30, 40, 60],
            "MonthlyCharges": [50.0, 70.0, 90.0, 60.0, 80.0, 110.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year"],
            "Churn": ["Yes", "No", "No", "Yes", "No", "No"],
        }
    )

    client = TestClient(app_module.app)
    response = client.post(
        "/analyze",
        files={"file": ("sample.csv", df_to_csv_bytes(df), "text/csv")},
        data={"analysis_type": "churn", "target_column": "Churn"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    run_id = payload["run_id"]

    stored = client.get(f"/runs/{run_id}")
    assert stored.status_code == 200
    data = stored.json()
    assert data["filename"] == "sample.csv"
    assert len(data["charts"]) > 0
    assert len(data["insights"]) > 0
    assert data["model_card"]["schema"]


def test_delete_removes_everything(monkeypatch):
    app_module, db_module = _reload_app(monkeypatch, cloudinary_enabled=True)

    monkeypatch.setattr(app_module, "upload_chart_png", lambda *args, **kwargs: {"secure_url": "https://example.com/chart.png", "public_id": "delete_me", "uploaded": True})

    df = pd.DataFrame({
        "tenure": [12, 24, 36, 48],
        "MonthlyCharges": [60.0, 75.0, 90.0, 80.0],
        "Contract": ["Month-to-month", "One year", "Two year", "One year"],
        "Churn": ["Yes", "No", "No", "Yes"],
    })

    client = TestClient(app_module.app)
    response = client.post(
        "/analyze",
        files={"file": ("delete_test.csv", df_to_csv_bytes(df), "text/csv")},
        data={"analysis_type": "churn", "target_column": "Churn"},
    )
    run_id = response.json()["run_id"]

    delete_response = client.delete(f"/runs/{run_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    get_response = client.get(f"/runs/{run_id}")
    assert get_response.status_code == 404
    assert len(db_module.get_runs()) == 0


def test_local_only_mode_works_without_keys(monkeypatch):
    app_module, db_module = _reload_app(monkeypatch, cloudinary_enabled=False)

    assert db_module.get_database_url().startswith("sqlite")
    client = TestClient(app_module.app)

    response = client.get("/health")
    assert response.status_code == 200

    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "sales": [100, 200, 300],
        "region": ["A", "B", "A"],
    })

    ana = client.post(
        "/analyze",
        files={"file": ("local_only.csv", df_to_csv_bytes(df), "text/csv")},
        data={"analysis_type": "general"},
    )
    assert ana.status_code == 200
    payload = ana.json()
    assert "inspection" in payload
    assert "model_results" in payload
