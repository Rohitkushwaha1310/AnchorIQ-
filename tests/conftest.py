"""
tests/conftest.py
Shared in-memory DataFrame fixtures and test clients for pytest.
"""
import io
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplconfig"))
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture for invoking endpoints."""
    return TestClient(app)


@pytest.fixture
def small_normal_df():
    """A standard small business dataset with continuous and categorical features and a binary target."""
    rng = np.random.RandomState(42)
    n = 60
    return pd.DataFrame({
        "tenure": rng.randint(1, 72, size=n),
        "MonthlyCharges": np.round(rng.uniform(20.0, 110.0, size=n), 2),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "Churn": rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6]),
    })


@pytest.fixture
def nulls_and_duplicates_df():
    """A dataset containing duplicate rows, missing numerical values, and missing categorical values."""
    base_rows = [
        {"tenure": 12, "MonthlyCharges": 50.0, "Contract": "Month-to-month", "Churn": "No"},
        {"tenure": 12, "MonthlyCharges": 50.0, "Contract": "Month-to-month", "Churn": "No"},  # Duplicate
        {"tenure": np.nan, "MonthlyCharges": 75.0, "Contract": "One year", "Churn": "Yes"},   # Missing numeric
        {"tenure": 24, "MonthlyCharges": np.nan, "Contract": None, "Churn": "No"},             # Missing numeric & category
        {"tenure": 36, "MonthlyCharges": 85.0, "Contract": "Two year", "Churn": "No"},
    ] * 10
    return pd.DataFrame(base_rows)


@pytest.fixture
def empty_df():
    """An empty DataFrame with no rows and no columns."""
    return pd.DataFrame()


@pytest.fixture
def one_column_df():
    """A single-column dataset (cannot train multi-feature predictive models)."""
    return pd.DataFrame({"Revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]})


@pytest.fixture
def single_unique_target_df():
    """A dataset where the target column contains only a single unique class (zero variance)."""
    rng = np.random.RandomState(42)
    n = 40
    return pd.DataFrame({
        "tenure": rng.randint(1, 48, size=n),
        "MonthlyCharges": np.round(rng.uniform(25.0, 95.0, size=n), 2),
        "Contract": rng.choice(["Month-to-month", "One year"], size=n),
        "Churn": ["No"] * n,  # Only one unique class
    })


@pytest.fixture
def fewer_than_30_rows_df():
    """A small dataset with fewer than 30 rows (e.g. 18 rows)."""
    rng = np.random.RandomState(7)
    n = 18
    return pd.DataFrame({
        "tenure": rng.randint(1, 36, size=n),
        "MonthlyCharges": np.round(rng.uniform(30.0, 80.0, size=n), 2),
        "Contract": rng.choice(["Month-to-month", "One year"], size=n),
        "Churn": rng.choice(["Yes", "No"], size=n),
    })


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Helper to convert an in-memory DataFrame to UTF-8 CSV bytes."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
