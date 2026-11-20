# AnchorIQ — Project Cleanup Inventory (Phase 0)

> Generated as part of the clean repository preparation.  
> **Status:** Inventory only. No files have been deleted, moved, or altered.

---

## 1. Executive Summary

| Category | Item Count | Total Size (Bytes) | Total Size (MB) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **GENERATED** | 508 | 170,998,348 | ~163.08 MB | Runtime output: charts, pickled models, schemas, uploaded CSVs, PDF reports, `__pycache__`, `.pytest_cache`, stray `.png` files |
| **DATA** | 5 | 1,037,026 | ~0.99 MB | Dataset files: `telco_churn.csv` and test fixture datasets in `tests/_generated_data/` |
| **SOURCE** | 53 | 438,421 | ~0.42 MB | Active application code, services, UI components, tests, configurations, docs, and legacy standalone scripts |
| **UNSURE** | 32 | 794,228 | ~0.76 MB | `anchoriq_local.db` (local fallback DB with 49 old test runs), `.mplconfig` (matplotlib font cache), and directory containers |
| **TOTAL** | **598** | **173,268,023** | **~165.25 MB** | *(Excludes `.git/` and `.venv/`)* |

---

## 2. Code Reference Audit (File & Folder Operations)

Audit of all code locations in `app/`, `services/`, `database/`, `ui/`, `tests/`, and root scripts performing file I/O:

| Location | Path / Target Referenced | Operation | Behavior: On-Demand vs Pre-existing Requirement | Safe to Delete Target? |
| :--- | :--- | :--- | :--- | :--- |
| `app/main.py:66-67` | `charts/`, `models/`, `reports/`, `uploads/` | `os.makedirs(folder, exist_ok=True)` | **Creates on demand** at app startup. | Directory can be emptied; directory itself should be kept with `.gitkeep`. |
| `app/main.py:72` | `charts/` | `app.mount("/charts", StaticFiles(directory="charts"), name="charts")` | **Requires directory to exist** at FastAPI startup. If `charts/` folder is completely missing, Starlette raises `RuntimeError`. | **Keep empty folder with `.gitkeep`**. Safe to delete all contents. |
| `app/main.py:126` | `uploads/<session_id>.csv` | `with open(path, "wb") as f:` | **Creates on demand** when an upload occurs. | Safe to delete contents of `uploads/`. |
| `app/main.py:181` | `models/` | `load_model_bundle(identifier, models_dir="models")` | **Reads existing model bundle** for a specific `session_id`. | Safe to delete old session models; new models are generated during runs. |
| `services/eda.py:18,55,73,102` | `charts/<session_id>/fig_*.png` | `os.makedirs`, `plt.savefig` | **Creates on demand** inside `auto_eda`. | Safe to delete all chart session folders in `charts/`. |
| `services/modeling.py:198,670,678`| `models/<session_id>_model.pkl`, `models/<session_id>_schema.json` | `os.makedirs`, `joblib.dump`, `open(..., "w")` | **Creates on demand** during `auto_model`. | Safe to delete all session `.pkl` and `.json` files in `models/`. |
| `services/explanations.py:37,92` | `charts/<session_id>/09_shap_summary.png` | `os.makedirs`, `plt.savefig` | **Creates on demand** during SHAP evaluation. | Safe to delete. |
| `services/pdf_report.py:73` | `reports/AnchorIQ_*.pdf` | `os.makedirs`, `doc.build` | **Creates on demand** during PDF generation. | Safe to delete all generated PDFs in `reports/`. |
| `services/pipeline.py:31,87` | `reports/<session_id>_enriched.csv` | `os.makedirs`, `out.to_csv` | **Creates on demand** during pipeline execution. | Safe to delete all enriched `.csv` files in `reports/`. |
| `services/predictor.py:76-84` | `models/<id>_model.pkl`, `models/<id>.pkl`, `models/model.pkl` | `joblib.load(path)` | **Reads model file** for the specified `session_id`. | Safe to delete old session models; runs create their own model on demand. |
| `database/db.py:148` | `./anchoriq_local.db` | `path.parent.mkdir`, SQLite connection | **Creates on demand** if emergency mode is active and file does not exist. | Safe to reset/clean. |
| `streamlit_app.py:297` | `charts/<session_id>` | `auto_eda(..., save_dir=chart_dir)` | **Creates on demand** during analysis. | Safe to delete. |
| `streamlit_app.py:657` | `reports/AnchorIQ_{type}_{id}.pdf` | `generate_pdf_report(..., output_path=pdf_path)` | **Creates on demand** on download request. | Safe to delete. |
| `tests/debug_api.py:8` | `telco_churn.csv` | `open("telco_churn.csv", "rb")` | **Requires pre-existing file** at root `telco_churn.csv`. | **DO NOT DELETE**. |
| `tests/smoke_test.py:14-16` | `telco_churn.csv`, `tests/_out` | `open("telco_churn.csv")`, `run_analysis(..., base_dir="tests/_out")` | `telco_churn.csv` required; `tests/_out` created on demand. | `telco_churn.csv` must remain accessible; `tests/_out` contents safe to delete. |
| `tests/test_api_and_services.py:160` | `telco_churn.csv` | `open("telco_churn.csv", "rb")` | **Requires pre-existing file** at root `telco_churn.csv`. | **DO NOT DELETE**. |
| `tests/test_full_suite.py:53-56` | `telco_churn.csv` | `assert os.path.exists(...)`, `open("telco_churn.csv", "rb")` | **Requires pre-existing file** at root `telco_churn.csv`. Explicitly asserts `os.path.exists("telco_churn.csv")`. | **DO NOT DELETE** from root unless test assertion path is updated. |
| `tests/test_step4.py:148-149` | `telco_churn.csv` | `assert os.path.exists("telco_churn.csv")` | **Requires pre-existing file** at root `telco_churn.csv`. Explicitly asserts `os.path.exists("telco_churn.csv")`. | **DO NOT DELETE** from root unless test assertion path is updated. |
| `churn_model.py:147,361,451` | `feature_analysis.png`, `model_evaluation.png`, `threshold_tuning.png` | `plt.savefig(...)` | Legacy output from running standalone script. | Safe to delete images. Standalone script moved to `legacy/`. |
| `report.py:13,69` | `reports/churn_with_predictions.csv`, `reports/AnchorIQ_Churn_Report.pdf` | `pd.read_csv`, ReportLab output | Legacy standalone script. | Safe to delete output files. Script moved to `legacy/`. |

---

## 3. Detailed File-by-File Classification

### A. GENERATED Artifacts (Confirmed Safe to Delete in Phase 1)

#### 1. Stray Root Images & Temporary DBs
- `feature_analysis.png` (111,132 bytes) — Generated by legacy `churn_model.py`. Not referenced in app, services, tests, or README.
- `model_evaluation.png` (149,433 bytes) — Generated by legacy `churn_model.py`. Not referenced in app, services, tests, or README.
- `test_persistence.sqlite` (86,016 bytes) — Temporary SQLite file created during persistence unit testing.

#### 2. Generated Charts (`charts/` — 20 subdirectories, 90 files, ~3.46 MB)
- `charts/17fab1d6/*`
- `charts/1ab327e2/*`
- `charts/1e9d3a37/*`
- `charts/23bfea32/*`
- `charts/2cee3e2f/*`
- `charts/2e17df4b/*`
- `charts/448bf929/*`
- `charts/48a2ed33/*`
- `charts/758ec71a/*`
- `charts/759d110a/*`
- `charts/8ad306fa/*`
- `charts/91dc5d85/*`
- `charts/a3195917/*`
- `charts/a8e2d20d/*`
- `charts/c3f3f5d5/*`
- `charts/d0360e6f/*`
- `charts/d104fbee/*`
- `charts/db073fb1/*`
- `charts/e554c321/*`
- `charts/f6e1c27f/*`
- `charts/fda600a3/*`
*(Note: Maintain directory `charts/` with a `.gitkeep` file so FastAPI `StaticFiles(directory="charts")` does not fail).*

#### 3. Uploaded Files (`uploads/` — 3 files, ~0.93 MB)
- `uploads/91dc5d85.csv` (979,010 bytes)
- `uploads/a8e2d20d.csv` (53 bytes)
- `uploads/f6e1c27f.csv` (457 bytes)
*(Note: Maintain directory `uploads/` with a `.gitkeep` file).*

#### 4. Reports & Predictions (`reports/` — 49 files, ~33.93 MB)
- All PDF files: `AnchorIQ_auto_*.pdf`, `AnchorIQ_churn_*.pdf`, `AnchorIQ_Churn_Report.pdf`, `AnchorIQ_d0be3050_report.pdf`, `AnchorIQ_e4a84d0b_report.pdf`, `AnchorIQ_f26eeec6_report.pdf`, `test_bullet.pdf`, `test_full.pdf`, `test_rep.pdf`, `test_report.pdf`, `test_v2.pdf`
- All enriched and prediction CSV files: `*_enriched.csv`, `*_new_predictions.csv`, `churn_with_predictions.csv`, `predictions.csv`
*(Note: Maintain directory `reports/` with a `.gitkeep` file).*

#### 5. Pickled Models & Schema JSONs (`models/` — 246 files, ~120.42 MB)
- All session model pickle files: `*_model.pkl`, `*.pkl` (including large runs like `057a6154_model.pkl` 44MB, `4bc06a4d_model.pkl` 44MB, `default_model.pkl` 13.5MB).
- All session schema JSON files: `*_schema.json` (including `default_schema.json` 6.6MB generated by `auto_model(session_id="default")`).
*(Note: Maintain directory `models/` with a `.gitkeep` file).*

#### 6. Test Output Artifacts (`tests/_out/` — 3 subdirectories, 18 files, ~3.87 MB)
- `tests/_out/charts/*`
- `tests/_out/models/*`
- `tests/_out/reports/*`
*(Note: Generated by `smoke_test.py`; safe to clean).*

#### 7. Cache Folders
- Root `__pycache__/` (7 files, ~0.12 MB)
- Subdirectory `app/__pycache__/`, `database/__pycache__/`, `services/__pycache__/`, `tests/__pycache__/`, `ui/__pycache__/`
- `.pytest_cache/` (5 files, ~0.01 MB)

---

### B. DATA Files (Do NOT Delete)

| File Path | Size | Reason / Usage |
| :--- | :--- | :--- |
| `telco_churn.csv` | 979,010 bytes | Primary baseline demo dataset. **Directly required** by `tests/test_step4.py`, `tests/test_full_suite.py`, `tests/test_api_and_services.py`, and `tests/debug_api.py`. |
| `tests/_generated_data/sales_report.xlsx` | 11,384 bytes | Test fixture for Excel format loading in test suite. |
| `tests/_generated_data/service_tickets.json`| 25,712 bytes | Test fixture for JSON format loading in test suite. |
| `tests/_generated_data/company.db` | 16,384 bytes | Test fixture for SQLite database ingestion in test suite. |
| `tests/_generated_data/table_data.pdf` | 4,546 bytes | Test fixture for PDF table extraction in test suite. |

---

### C. SOURCE Code, Config & Documentation (Do NOT Delete)

#### 1. Core Application & API
- `app/__init__.py`
- `app/main.py` — Primary FastAPI backend application.
- `streamlit_app.py` — Primary Streamlit frontend application.

#### 2. Services
- `services/__init__.py`
- `services/ab_testing.py` — Hypothesis testing service.
- `services/agent.py` — Chat & evidence assistant service.
- `services/cleaning.py` — Data imputation and preprocessing.
- `services/cloudinary.py` — Cloudinary image & artifact storage.
- `services/eda.py` — Automated visualization engine.
- `services/explanations.py` — SHAP & feature importance engine.
- `services/insights.py` — Groq AI / rule-based executive insights.
- `services/inspection.py` — Dataset profiling & heuristic target selection.
- `services/loader.py` — Multi-format file parsing.
- `services/modeling.py` — Tournament AutoML engine.
- `services/pdf_report.py` — ReportLab executive PDF generator.
- `services/pipeline.py` — Unified legacy pipeline interface.
- `services/predictor.py` — Model inference & calibrated scoring.

#### 3. Database
- `database/__init__.py`
- `database/db.py` — Supabase PostgreSQL & SQLite engine, ORM models, session manager.
- `database/schema.sql` — Supabase PostgreSQL schema definition DDL.

#### 4. UI System
- `ui/__init__.py`
- `ui/style.py` — Design tokens, CSS injection, layout component helpers.
- `.streamlit/config.toml` — Streamlit theme and server configuration.

#### 5. Test Suite
- `tests/conftest.py`
- `tests/test_step4.py` — Primary regression & integration test suite.
- `tests/test_api_and_services.py` — API routes & services unit tests.
- `tests/test_full_suite.py` — Full suite integration tests.
- `tests/test_persistence.py` — Database & storage persistence tests.
- `tests/test_agent.py` — Assistant chat test.
- `tests/smoke_test.py` — Smoke tests.
- `tests/debug_api.py` — Manual endpoint debug script.

#### 6. Configuration & Documentation
- `.env` — Secrets & configuration keys (Supabase, Groq, Cloudinary). **NEVER DELETE OR PRINT**.
- `.env.example` — Environment template.
- `.gitignore` — Git ignore list.
- `pytest.ini` — Pytest configuration.
- `README.md` — Project documentation.
- `requirements.txt` — Python dependency manifest.
- `docs/ui_connection_map.md` — UI connection map generated in Phase 0.
- `docs/cleanup_inventory.md` — This cleanup inventory document.

#### 7. Legacy Standalone Scripts (To Be Moved to `legacy/` in Phase 2)
- `churn_model.py` (16,544 bytes) — Standalone training script from August 2026.
- `main.py` (5,469 bytes) — Standalone exploratory script from July 2026.
- `report.py` (9,034 bytes) — Standalone ReportLab script from August 2026.
- `apply_streamlit_patch.py` (6,541 bytes) — Standalone patcher script.

---

### D. UNSURE Items (Requires User Decision or Special Handling)

| Item Path | Type | Size | Description & Recommendation |
| :--- | :--- | :--- | :--- |
| `anchoriq_local.db` | FILE | 544,768 bytes | SQLite database containing 49 old run records from historical local test sessions. Supabase is the primary database. If deleted, `database/db.py` automatically creates a fresh, empty SQLite file on demand. **Recommendation**: Reset/remove so the local database starts brand new with 0 old runs. |
| `.mplconfig/` | DIR | 243,923 bytes | Matplotlib local font cache directory containing `fontlist-v*.json`. Auto-generated by matplotlib at runtime. **Recommendation**: Safe to delete; matplotlib re-generates on demand. |

---

## 4. Phase 1 Deletion Plan & Safety Assertions

1. **Safety Guarantee**:
   - `charts/`, `models/`, `reports/`, and `uploads/` will have their files deleted, but their directory containers will be retained with `.gitkeep` to satisfy `app/main.py:72` (`StaticFiles(directory="charts")`).
   - `telco_churn.csv` will **NOT** be deleted.
   - `.env` will **NOT** be touched.
   - All source code in `app/`, `services/`, `database/`, `ui/`, `tests/` will **NOT** be modified in Phase 1.
2. **Phase 2 Plan (Tidy)**:
   - Create `legacy/` and move `churn_model.py`, `main.py`, `report.py`, `apply_streamlit_patch.py`.
   - Keep `telco_churn.csv` at project root (or create `samples/telco_churn.csv` and add a fallback symlink/copy so existing tests in `test_step4.py` and `test_full_suite.py` asserting `os.path.exists("telco_churn.csv")` continue to pass 100%).

