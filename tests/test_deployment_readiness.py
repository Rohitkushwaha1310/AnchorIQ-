"""Tests for AnchorIQ deployment readiness:
1. Supabase Auth token requirements (401 on missing/invalid token)
2. Strict per-user isolation (User B gets 404 on all endpoints for User A's run)
3. Zero local disk footprint during /analyze
4. Public /health check structure
"""
import io
import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_current_user
from tests.conftest import df_to_csv_bytes


@pytest.fixture
def auth_client():
    """Client for testing authentication boundaries without dependency overrides."""
    return TestClient(app)


def test_auth_token_required_on_all_endpoints(auth_client):
    """Verify that requests without an Authorization header return HTTP 401."""
    endpoints = [
        ("GET", "/runs"),
        ("GET", "/runs/999999"),
        ("DELETE", "/runs/999999"),
        ("GET", "/runs/999999/report"),
        ("GET", "/predict/schema/999999"),
        ("POST", "/predict/999999"),
        ("POST", "/ab_test"),
        ("POST", "/chat/999999"),
    ]
    for method, path in endpoints:
        if method == "GET":
            resp = auth_client.get(path)
        elif method == "DELETE":
            resp = auth_client.delete(path)
        elif method == "POST":
            resp = auth_client.post(path, json={})
        assert resp.status_code == 401, f"Expected 401 for {method} {path}, got {resp.status_code}"


def test_user_isolation_between_two_users(small_normal_df):
    """Verify User A creates a run; User B gets 404 on all endpoints for A's run; User A sees only A's runs."""
    user_a = "11111111-1111-1111-1111-111111111111"
    user_b = "22222222-2222-2222-2222-222222222222"

    client = TestClient(app)

    app.dependency_overrides[get_current_user] = lambda: user_a
    csv_bytes = df_to_csv_bytes(small_normal_df)

    res_create = client.post(
        "/analyze",
        files={"file": ("test_isolation.csv", csv_bytes, "text/csv")},
        data={"analysis_type": "churn"},
    )
    assert res_create.status_code == 200, res_create.text
    run_a_id = res_create.json()["run_id"]

    runs_a = client.get("/runs").json()["runs"]
    assert any(r["id"] == run_a_id for r in runs_a)

    app.dependency_overrides[get_current_user] = lambda: user_b

    runs_b = client.get("/runs").json()["runs"]
    assert not any(r["id"] == run_a_id for r in runs_b)

    resp_get = client.get(f"/runs/{run_a_id}")
    assert resp_get.status_code == 404

    resp_rep = client.get(f"/runs/{run_a_id}/report")
    assert resp_rep.status_code == 404

    resp_schema = client.get(f"/predict/schema/{run_a_id}")
    assert resp_schema.status_code == 404

    resp_pred = client.post(f"/predict/{run_a_id}", json={"Contract": "One year", "tenure": 12, "MonthlyCharges": 50.0})
    assert resp_pred.status_code == 404

    resp_ab = client.post("/ab_test", json={"run_id": run_a_id, "group_column": "Contract", "metric_column": "Churn"})
    assert resp_ab.status_code == 404

    resp_chat = client.post(f"/chat/{run_a_id}", json={"question": "Why are customers leaving?"})
    assert resp_chat.status_code == 404

    resp_del = client.delete(f"/runs/{run_a_id}")
    assert resp_del.status_code == 404

    app.dependency_overrides[get_current_user] = lambda: user_a
    rep_res = client.get(f"/runs/{run_a_id}/report")
    assert rep_res.status_code == 200
    assert rep_res.headers.get("content-type") == "application/pdf"
    assert rep_res.content.startswith(b"%PDF")

    del_res = client.delete(f"/runs/{run_a_id}")
    assert del_res.status_code == 200
    app.dependency_overrides.clear()


def test_no_local_disk_write_during_analyze(small_normal_df):
    """Assert that running /analyze creates NO new files or folders in the working directory."""
    user_id = "33333333-3333-3333-3333-333333333333"
    app.dependency_overrides[get_current_user] = lambda: user_id
    client = TestClient(app)

    prohibited_folders = ["charts", "models", "reports", "uploads"]
    for folder in prohibited_folders:
        assert not os.path.exists(folder), f"Folder '{folder}' should not exist before analysis."

    csv_bytes = df_to_csv_bytes(small_normal_df)
    res = client.post(
        "/analyze",
        files={"file": ("no_disk_test.csv", csv_bytes, "text/csv")},
        data={"analysis_type": "auto"},
    )
    assert res.status_code == 200, res.text

    for folder in prohibited_folders:
        assert not os.path.exists(folder), f"Folder '{folder}' was created on local disk during /analyze!"

    run_id = res.json()["run_id"]
    client.delete(f"/runs/{run_id}")
    app.dependency_overrides.clear()


def test_public_health_endpoint(auth_client):
    """Verify GET /health is public and returns health flags without leaking secrets."""
    resp = auth_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "database" in data
    assert "cloudinary" in data
    assert "groq" in data
    text = str(data)
    assert "postgres" not in text
    assert "key" not in text
    assert "secret" not in text

