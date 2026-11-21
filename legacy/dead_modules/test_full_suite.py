"""
Comprehensive End-to-End Test Suite for AnchorIQ
Tests all file types, HTTP endpoints, edge cases, error conditions, and quality requirements.
"""
import io
import json
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import reportlab.lib.pagesizes
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Add project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.main import app
from services.loader import load_any
from services.pipeline import run_analysis

client = TestClient(app)
TEST_DIR = os.path.join(ROOT, "tests", "_generated_data")
os.makedirs(TEST_DIR, exist_ok=True)

passed = 0
failed = 0

def assert_test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"[PASS] {name}")
        passed += 1
    else:
        print(f"[FAIL] {name} - {detail}")
        failed += 1


def run_all_tests():
    print("==================================================================")
    print("      ANCHORIQ COMPREHENSIVE TEST SUITE (STEP 3, 4, 5)            ")
    print("==================================================================")

    # ──────────────────────────────────────────────────────────────────
    # 1. TEST TELCO CHURN (Binary Classification Target)
    # ──────────────────────────────────────────────────────────────────
    print("\n--- 1. Telco Churn Dataset (Binary Text Target) ---")
    telco_path = os.path.join(ROOT, "telco_churn.csv")
    assert_test("telco_churn.csv exists", os.path.exists(telco_path))

    with open(telco_path, "rb") as f:
        content = f.read()

    resp = client.post("/analyze", files={"file": ("telco_churn.csv", content, "text/csv")}, data={"analysis_type": "churn"})
    assert_test("POST /analyze telco_churn returns 200", resp.status_code == 200, resp.text)
    res = resp.json()
    session_id = res["session_id"]
    assert_test("Target auto-detected as Churn", res["profile"]["target"] == "Churn")
    assert_test("Problem type is classification", res["profile"]["problem_type"] == "classification")
    assert_test("Model trained with AUC > 0.75", res["model_results"].get("auc", 0) > 0.75)
    assert_test("Accuracy metric present", "accuracy" in res["model_results"])
    assert_test("Feature importance present", len(res["model_results"].get("feature_importance", {})) > 0)
    assert_test("Plotly charts generated", len(res.get("plotly_charts", {})) >= 3)
    assert_test("Interactive charts include gains and risk tiers", "gains" in res["plotly_charts"] and "risk_tiers" in res["plotly_charts"])
    assert_test("Recommendations contain numbers", any(any(char.isdigit() for char in r["action"] + r["impact"]) for r in res.get("recommendations", [])))

    # Test PDF generation endpoint
    resp_pdf = client.post("/generate_report", files={"file": ("telco_churn.csv", content, "text/csv")}, data={"analysis_type": "churn"})
    assert_test("POST /generate_report returns 200", resp_pdf.status_code == 200)
    assert_test("POST /generate_report returns valid PDF bytes", resp_pdf.content.startswith(b"%PDF"))

    # Test Download Enriched CSV endpoint
    resp_dl = client.get(f"/download/{session_id}")
    assert_test(f"GET /download/{session_id} returns 200", resp_dl.status_code == 200)
    df_enriched = pd.read_csv(io.BytesIO(resp_dl.content))
    assert_test("Enriched CSV has prediction_score", "prediction_score" in df_enriched.columns)
    assert_test("Enriched CSV has risk_tier", "risk_tier" in df_enriched.columns)
    assert_test("Enriched CSV has predicted_Churn", "predicted_Churn" in df_enriched.columns)

    # Test Predict with session model on new data
    new_data = df_enriched.drop(columns=["Churn", "predicted_Churn", "prediction_score", "risk_tier"], errors="ignore").head(20)
    new_csv = new_data.to_csv(index=False).encode("utf-8")
    resp_pred = client.post(f"/predict/{session_id}", files={"file": ("new_customers.csv", new_csv, "text/csv")})
    assert_test(f"POST /predict/{session_id} returns 200", resp_pred.status_code == 200, resp_pred.text)
    df_new_pred = pd.read_csv(io.BytesIO(resp_pred.content))
    assert_test("New predictions contain predicted_Churn", "predicted_Churn" in df_new_pred.columns)
    assert_test("New predictions contain prediction_score", "prediction_score" in df_new_pred.columns)
    assert_test("New predictions contain risk_tier", "risk_tier" in df_new_pred.columns)

    # ──────────────────────────────────────────────────────────────────
    # 2. TEST SALES EXCEL FILE (.xlsx) WITH TITLE ROW & OUTLIER (No Target)
    # ──────────────────────────────────────────────────────────────────
    print("\n--- 2. Sales Excel (.xlsx) with Title Row & Outlier (No Target) ---")
    sales_path = os.path.join(TEST_DIR, "sales_report.xlsx")
    n_sales = 200
    dates = pd.date_range("2024-01-01", periods=n_sales, freq="D")
    rng = np.random.RandomState(42)
    regions = rng.choice(["North", "South", "East", "West"], size=n_sales)
    units = rng.poisson(lam=25, size=n_sales)
    revenue = units * rng.uniform(80, 120, size=n_sales)
    revenue[50] = 500000.0  # Extreme outlier

    df_sales_raw = pd.DataFrame({
        "Date": dates,
        "Region": regions,
        "Units": units,
        "Revenue": revenue
    })

    # Create Excel file with 2 metadata/title rows above actual table
    with pd.ExcelWriter(sales_path, engine="openpyxl") as writer:
        title_df = pd.DataFrame([["ACME GLOBAL SALES REPORT", ""], ["CONFIDENTIAL - FOR INTERNAL USE ONLY", ""]])
        title_df.to_excel(writer, sheet_name="SalesData", index=False, header=False)
        df_sales_raw.to_excel(writer, sheet_name="SalesData", startrow=3, index=False)

    with open(sales_path, "rb") as f:
        excel_content = f.read()

    resp_excel = client.post("/analyze", files={"file": ("sales_report.xlsx", excel_content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert_test("POST /analyze sales.xlsx returns 200", resp_excel.status_code == 200, resp_excel.text)
    res_excel = resp_excel.json()
    assert_test("Excel header detected past title row", "Revenue" in res_excel["preview"][0])
    assert_test("No target detected -> Unsupervised mode triggered", res_excel["profile"]["target"] is None)
    assert_test("Segments generated", res_excel["segments"] is not None)
    assert_test("Anomalies generated", res_excel["anomalies"] is not None)
    assert_test("Extreme outlier flagged", res_excel["anomalies"]["extreme_count"] >= 1)
    assert_test("Time-series forecast generated", res_excel["forecast"] is not None)
    assert_test("Forecast metric is Revenue", res_excel["forecast"]["metric"] == "Revenue")

    # ──────────────────────────────────────────────────────────────────
    # 3. TEST NESTED JSON WITH MULTI-CLASS TARGET
    # ──────────────────────────────────────────────────────────────────
    print("\n--- 3. Nested JSON with Multi-Class Target ---")
    json_path = os.path.join(TEST_DIR, "service_tickets.json")
    n_tickets = 150
    priorities = rng.choice(["Low", "Medium", "High", "Critical"], size=n_tickets)
    nested_data = {
        "metadata": {"system": "HelpDesk v4", "version": "1.0"},
        "payload": {
            "records": [
                {
                    "ticket_id": f"TICK-{1000+i}",
                    "user": {"tier": rng.choice(["Standard", "Gold", "Platinum"]), "tenure_months": int(rng.randint(1, 48))},
                    "metrics": {"response_time_min": float(rng.exponential(30)), "reopen_count": int(rng.poisson(1))},
                    "priority": priorities[i]
                }
                for i in range(n_tickets)
            ]
        }
    }
    with open(json_path, "w") as f:
        json.dump(nested_data, f)

    with open(json_path, "rb") as f:
        json_content = f.read()

    resp_json = client.post("/analyze", files={"file": ("service_tickets.json", json_content, "application/json")}, data={"target_column": "priority"})
    assert_test("POST /analyze nested.json returns 200", resp_json.status_code == 200, resp_json.text)
    res_json = resp_json.json()
    assert_test("Nested JSON flattened successfully", "user.tier" in res_json["preview"][0] or "metrics.response_time_min" in res_json["preview"][0])
    assert_test("Multi-class classification trained", res_json["profile"]["problem_type"] == "classification")
    assert_test("Multi-class accuracy present", res_json["model_results"].get("accuracy") is not None)

    # ──────────────────────────────────────────────────────────────────
    # 4. TEST FORMATS: JSONL, TSV, SEMICOLON CSV, PARQUET, XML, SQLITE
    # ──────────────────────────────────────────────────────────────────
    print("\n--- 4. Other File Formats (JSONL, TSV, Semicolon CSV, Parquet, XML, SQLite) ---")
    df_sample = pd.DataFrame({
        "id": range(1, 101),
        "department": rng.choice(["Engineering", "Sales", "Support", "Marketing"], 100),
        "experience": rng.uniform(1, 15, 100).round(1),
        "salary": rng.uniform(50000, 140000, 100).round(2),
        "performance": rng.choice(["Meets", "Exceeds", "Needs Improvement"], 100)
    })

    # a) JSONL
    jsonl_bytes = "\n".join(json.dumps(row) for row in df_sample.to_dict("records")).encode("utf-8")
    resp_jsonl = client.post("/analyze", files={"file": ("employees.jsonl", jsonl_bytes, "application/x-ndjson")})
    assert_test("JSONL parsed and analyzed", resp_jsonl.status_code == 200)

    # b) TSV
    tsv_bytes = df_sample.to_csv(sep="\t", index=False).encode("utf-8")
    resp_tsv = client.post("/analyze", files={"file": ("employees.tsv", tsv_bytes, "text/tab-separated-values")})
    assert_test("TSV parsed and analyzed", resp_tsv.status_code == 200)

    # c) Semicolon-separated CSV
    semi_bytes = df_sample.to_csv(sep=";", index=False).encode("utf-8")
    resp_semi = client.post("/analyze", files={"file": ("employees_semi.csv", semi_bytes, "text/csv")})
    assert_test("Semicolon CSV parsed and analyzed", resp_semi.status_code == 200)

    # d) Parquet
    buf_pq = io.BytesIO()
    df_sample.to_parquet(buf_pq, index=False)
    resp_pq = client.post("/analyze", files={"file": ("employees.parquet", buf_pq.getvalue(), "application/octet-stream")})
    assert_test("Parquet parsed and analyzed", resp_pq.status_code == 200)

    # e) XML
    xml_root = ET.Element("employees")
    for r in df_sample.to_dict("records"):
        item = ET.SubElement(xml_root, "employee")
        for k, v in r.items():
            child = ET.SubElement(item, str(k))
            child.text = str(v)
    xml_bytes = ET.tostring(xml_root, encoding="utf-8")
    resp_xml = client.post("/analyze", files={"file": ("employees.xml", xml_bytes, "application/xml")})
    assert_test("XML parsed and analyzed", resp_xml.status_code == 200)

    # f) SQLite
    sqlite_path = os.path.join(TEST_DIR, "company.db")
    if os.path.exists(sqlite_path): os.unlink(sqlite_path)
    con = sqlite3.connect(sqlite_path)
    df_sample.to_sql("staff", con, index=False)
    con.close()
    with open(sqlite_path, "rb") as f:
        sqlite_bytes = f.read()
    resp_sqlite = client.post("/analyze", files={"file": ("company.db", sqlite_bytes, "application/x-sqlite3")})
    assert_test("SQLite .db parsed and analyzed", resp_sqlite.status_code == 200)

    # ──────────────────────────────────────────────────────────────────
    # 5. TEST PDF WITH EMBEDDED TABLE
    # ──────────────────────────────────────────────────────────────────
    print("\n--- 5. PDF Containing a Table ---")
    pdf_tbl_path = os.path.join(TEST_DIR, "table_data.pdf")
    from reportlab.lib import colors
    doc = SimpleDocTemplate(pdf_tbl_path, pagesize=reportlab.lib.pagesizes.letter)
    table_data = [["CustID", "Region", "MonthlySpend", "ContractYears", "Active"]]
    for i in range(1, 60):
        table_data.append([str(1000 + i), str(rng.choice(["North", "South", "East"])), f"{rng.uniform(30, 200):.2f}", str(rng.randint(1, 5)), str(rng.choice(["Yes", "No"]))])
    tbl_flow = Table(table_data)
    tbl_flow.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333333")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#cccccc")),
    ]))
    elements = [Paragraph("Customer Portfolio Summary", getSampleStyleSheet()["Heading1"]), tbl_flow]
    doc.build(elements)

    with open(pdf_tbl_path, "rb") as f:
        pdf_table_bytes = f.read()
    resp_pdf_tbl = client.post("/analyze", files={"file": ("table_data.pdf", pdf_table_bytes, "application/pdf")})
    assert_test("PDF Table extraction and analysis", resp_pdf_tbl.status_code == 200, resp_pdf_tbl.text)

    # ──────────────────────────────────────────────────────────────────
    # 6. TEST BAD INPUTS (Must return HTTP 400, never 500)
    # ──────────────────────────────────────────────────────────────────
    print("\n--- 6. Bad Inputs Testing (Validation for HTTP 400) ---")

    # a) Empty file
    resp_empty = client.post("/analyze", files={"file": ("empty.csv", b"", "text/csv")})
    assert_test("Empty file returns 400", resp_empty.status_code == 400 and "empty" in resp_empty.json()["detail"].lower())

    # b) One-column file
    one_col = "Values\n1\n2\n3\n4\n5".encode()
    resp_1col = client.post("/analyze", files={"file": ("one_col.csv", one_col, "text/csv")})
    assert_test("One-column file returns 400", resp_1col.status_code == 400 and "at least 2 columns" in resp_1col.json()["detail"].lower())

    # c) Unsupported extension
    resp_bad_ext = client.post("/analyze", files={"file": ("script.exe", b"binary content", "application/x-msdownload")})
    assert_test("Unsupported extension returns 400", resp_bad_ext.status_code == 400 and "unsupported" in resp_bad_ext.json()["detail"].lower())

    # d) 100% missing values
    all_nulls = "colA,colB,colC\n,,,\n,,,\n,,,".encode()
    resp_nulls = client.post("/analyze", files={"file": ("nulls.csv", all_nulls, "text/csv")})
    assert_test("100% missing values returns 400", resp_nulls.status_code == 400 and "missing values" in resp_nulls.json()["detail"].lower())

    # e) Invalid session ID on /download
    resp_bad_session = client.get("/download/invalid_id!123")
    assert_test("Invalid session ID format returns 400", resp_bad_session.status_code == 400)

    # ──────────────────────────────────────────────────────────────────
    # 7. REGRESSION TARGET & LEAKAGE DETECTION
    # ──────────────────────────────────────────────────────────────────
    print("\n--- 7. Regression Target & Target Leakage Detection ---")
    n_reg = 120
    sqft = rng.uniform(500, 3500, n_reg)
    bedrooms = rng.randint(1, 6, n_reg)
    price = (sqft * 250 + bedrooms * 15000 + rng.normal(0, 5000, n_reg)).round(2)
    leaking_price_copy = price.copy()  # Exact copy of target
    leaking_price_near = (price + rng.normal(0, 0.01, n_reg)).round(2)  # 0.999 correlation

    df_reg = pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "house_price": price,
        "price_duplicate": leaking_price_copy,
        "price_near_leak": leaking_price_near,
        "neighborhood": rng.choice(["Downtown", "Suburbs", "Uptown"], n_reg)
    })
    reg_csv = df_reg.to_csv(index=False).encode("utf-8")
    resp_reg = client.post("/analyze", files={"file": ("houses.csv", reg_csv, "text/csv")}, data={"target_column": "house_price"})
    assert_test("POST /analyze regression returns 200", resp_reg.status_code == 200, resp_reg.text)
    res_reg = resp_reg.json()
    assert_test("Problem type identified as regression", res_reg["profile"]["problem_type"] == "regression")
    assert_test("Regression R2 score computed", res_reg["model_results"].get("r2_score") is not None)
    assert_test("Regression RMSE computed", res_reg["model_results"].get("rmse") is not None)
    leaks = res_reg["model_results"].get("dropped_for_leakage", {})
    assert_test("Leakage detected and reported", len(leaks) >= 1)
    assert_test("Leaking columns dropped from features", "price_duplicate" not in res_reg["model_results"]["features_used"])

    # ──────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────
    print("\n==================================================================")
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("==================================================================")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
