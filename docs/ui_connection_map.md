# AnchorIQ UI Connection Map

This document establishes the baseline connection map for `streamlit_app.py` and its UI helper components prior to redesign. Every item listed here must be strictly preserved under the "no broken connection" contract.

---

## 1. Calls from UI to Backend & Internal Services

### A. FastAPI Backend REST Calls

| Endpoint | HTTP Method | Invocation Location in UI | Payload Fields Sent | Response Fields Read by UI | Fallback Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `{api_url}/health` | `GET` | Lines 254–258 (Engine auto-detection) | None | `response.status_code` (determines whether `use_api = True`) | Falls back to Direct Embedded Engine (`use_api = False`) |
| `{api_url}/analyze` | `POST` | Lines 270–276 (Run analysis via REST) | `files={"file": (name, bytes, "application/octet-stream")}`, `data={"analysis_type": str, "target_column": str}` | Full analysis payload: `error`, `inspection`, `model_results`, `cleaning`, `chart_paths`, `chart_urls`, `insights`, `run_id`, `session_id`, `filename` | Catches exceptions, displays warning, falls back to direct embedded engine |
| `{api_url}/runs` | `GET` | Top-bar run selector | None | List of runs: `[{"id": int, "filename": str, "created_at": str, ...}]` | Direct fallback to `database.db.get_runs()` |
| `{api_url}/runs/{run_id}` | `GET` | Top-bar run loader | None | Single run record: `{"id": int, "filename": str, "model_card": dict, "charts": list, "insights": list, ...}` | Direct fallback to `database.db.get_run(run_id)` |
| `{api_url}/ab_test` | `POST` | Lines 751–754 (Tab 7: A/B Testing) | `{"run_id": str/int, "group_column": str, "metric_column": str}` | `significant` (bool), `p_value` (float), `test_type` (str), `plain_english_verdict` (str), `effect_size_label` (str), `effect_size` (any), `confidence_interval` (list), `metric_type` (str), `group_stats` / `group_sizes` (dict), `warnings` (list), `notes` (list) | Falls back to in-memory `services.ab_testing.run_ab_test()` |
| `{api_url}/predict/schema/{run_id}` | `GET` | Lines 812–815 (Tab 8: Schema fetch) | None | `schema` dict: `columns` / `fields` with `kind`, `allowed_categories`, `min`, `max` | Falls back to `ml.get("schema")` from `st.session_state["result"]` |
| `{api_url}/predict/{run_id}` | `POST` | Lines 860–865 (Tab 8: Model inference) | `input_fields` (dict mapping feature column names to user input values) | `prediction` (str/number), `calibrated_probability` / `probability` (float), `risk_label` (str: Low/Medium/High), `recommendation` (str), `reasons` (list of str), `shap_status` (str) | Falls back to in-memory `services.predictor.predict_single()` |
| `{api_url}/chat/{run_id}` | `POST` | Lines 961–966 (Tab 9: Assistant chat) | `{"question": str, "inputs": Optional[dict]}` | `answer` (str) | Catches exception, appends failure note to chat history |

---

### B. Direct Embedded Services (In-Memory Engine)

When the FastAPI server is not selected or offline, the UI executes internal services directly with zero downtime:

| Service Function | Imported From | Arguments Provided by UI | Return Value Used by UI |
| :--- | :--- | :--- | :--- |
| `inspect_dataset()` | `services.inspection` | `df_raw: pd.DataFrame`, `analysis_type: str`, `target_column: Optional[str]` | `inspection` dict (`rows`, `columns`, `missing_total`, `duplicates`, `target_column`, `stats`, `numerical_cols`, `categorical_cols`, `missing_cols`) |
| `auto_clean()` | `services.cleaning` | `df_raw: pd.DataFrame` | Tuple: `(df_clean: pd.DataFrame, clean_report: dict)` with `original_shape`, `cleaned_shape`, `steps_applied` |
| `auto_eda()` | `services.eda` | `df_clean: pd.DataFrame`, `target: str`, `analysis_type: str`, `save_dir: str` | `chart_paths: list[str]` (paths to generated high-res PNG plots) |
| `auto_model()` | `services.modeling` | `df_clean: pd.DataFrame`, `target: str`, `analysis_type: str` | `model_results: dict` (`problem_type`, `best_model`, `accuracy`, `auc`, `r2`, `mae`, `cv_mean`, `cv_std`, `model_card`, `decision_rules`, `feature_importance`, `schema`) |
| `generate_insights()` | `services.insights` | `inspection: dict`, `model_results: dict`, `target: str`, `analysis_type: str`, `feature_importance: dict` | `insights: str` (executive business narrative in markdown) |
| `generate_pdf_report()` | `services.pdf_report` | `result: dict`, `chart_paths: list[str]`, `output_path: str` | Path to generated `.pdf` file on disk |
| `run_ab_test()` | `services.ab_testing` | `df: pd.DataFrame`, `group_column: str`, `metric_column: str` | `ab_result: dict` with statistical test outcomes |
| `predict_single()` | `services.predictor` | `session_id: str`, `input_data: dict` | `pred_res: dict` with predictions, probability, risk tier, and explanatory reasons |

---

## 2. Session State Keys (`st.session_state`)

| Key Name | Type | Where It Is Set | Where It Is Read | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `"result"` | `dict` | Line 331 (upon analysis completion) | Lines 347, 348 | Holds the entire analysis result payload for rendering all dashboard tabs |
| `"analysis_type"` | `str` | Line 332 (upon analysis completion) | Line 349 | Stores active business domain objective (`"auto"`, `"churn"`, etc.) |
| `"uploaded_file_bytes"` | `bytes` | Line 333 | Line 713 | Raw file content cached for re-analysis, PDF generation, or A/B test data parsing |
| `"uploaded_file_name"` | `str` | Line 334 | Line 714 | Original filename (e.g., `telco_churn.csv`) displayed in header and metadata |
| `"df_raw"` | `pd.DataFrame` | Line 244 (upon file upload) | Line 708 | Cached raw dataset before transformations; used by A/B testing |
| `"df_clean"` | `pd.DataFrame` | Line 336 (when direct engine cleans dataset) | Line 706 | Cached imputed/transformed dataset; used by A/B testing |
| `f"anchoriq_chat_{run_id_val}"` | `list[dict]` | Line 924 (`setdefault([])`), Lines 959, 970 (message appends) | Line 944 | Persists conversation history list `[{"role": "user"/"assistant", "content": str}]` across Streamlit re-renders |

---

## 3. Widget Keys (`key="..."`) & Dependent Logic

| Widget Key | Widget Type | Location in UI | Dependent Logic & Behavior |
| :--- | :--- | :--- | :--- |
| `"tab7_group_col"` | `st.selectbox` | Line 733 (Tab 7) | Supplies the chosen segmentation column for A/B testing payload (`group_column`) |
| `"tab7_metric_col"` | `st.selectbox` | Line 739 (Tab 7) | Supplies the chosen numerical or binary outcome column for A/B testing payload (`metric_column`) |
| `"tab7_run_btn"` | `st.button` | Line 742 (Tab 7) | Triggers execution of the hypothesis test computation (REST or in-memory) |
| `f"predict_form_{run_id_val}"` | `st.form` | Line 825 (Tab 8) | Groups dynamic feature inputs into an atomic form submission |
| `f"form_input_{col_name}"` | `st.selectbox` / `st.number_input` | Lines 840, 851 (Tab 8) | Captures feature input values dynamically according to column schema |
| `f"chat_suggestion_{run_id_val}_{index}"` | `st.button` | Line 934 (Tab 9) | Quick-prompt pill buttons; clicking one populates `selected_question` |
| `f"chat_inputs_{run_id_val}"` | `st.text_area` | Line 940 (Tab 9) | Optional customer feature JSON payload passed to chat endpoint for probability estimation |

---

## 4. File Paths, Artifacts & URLs Loaded

| Resource Type | Resource Path / URL Format | Where Loaded | Behavior & Fallback |
| :--- | :--- | :--- | :--- |
| Uploaded Dataset | User's file buffer via `st.file_uploader` | Lines 142–169 | Read into memory via `pd.read_csv`, `pd.read_excel`, or `pd.read_json` |
| Chart Images | `charts/{session_id}/fig_*.png` | Line 470 (Tab 1: EDA Charts) | Loaded via `st.image(p)` if `os.path.exists(p)` |
| Cloudinary Chart URLs | `https://res.cloudinary.com/...` | Returned in `res.get("chart_urls", [])` | When available, displayed in place of local paths |
| Executive PDF Report | `reports/AnchorIQ_{analysis_type}_{session_id}.pdf` | Lines 663–678 (Tab 6: PDF Report) | Built on demand via `generate_pdf_report()`, read as binary bytes, served via `st.download_button` |
| Serialized Model | `models/{session_id}_model.pkl` | Generated during `auto_model` | Stored locally for inferences |
| Schema Definition | `models/{session_id}_schema.json` | Generated during `auto_model` | Loaded for dynamic form generation if REST endpoint is unreachable |

---

## 5. Environment Variables Read

While `streamlit_app.py` avoids direct calls to `os.getenv` for secrets, it invokes service modules that consume the following environment variables:

| Environment Variable | Consumed In | Purpose | Fallback if Missing |
| :--- | :--- | :--- | :--- |
| `DATABASE_URL` / `SUPABASE_DB_URL` | `database/db.py` | Connection string to Supabase PostgreSQL | Activates emergency local SQLite database at `anchoriq_local.db` |
| `ANCHORIQ_EMERGENCY_DB_PATH` | `database/db.py` | Path for fallback SQLite database | Defaults to `./anchoriq_local.db` |
| `GROQ_API_KEY` | `services/insights.py` | Authentication key for Groq LLM inference | Falls back automatically to `_rule_based_insights()` |
| `GROQ_MODEL` | `services/insights.py` | Preferred model name on Groq | Defaults to `"openai/gpt-oss-120b"` with candidate fallback chain |
| `CLOUDINARY_CLOUD_NAME` | `services/cloudinary.py` | Cloudinary tenant account name | Fallback to local disk storage (`charts/<session_id>`) |
| `CLOUDINARY_API_KEY` | `services/cloudinary.py` | Cloudinary API Key | Fallback to local disk storage |
| `CLOUDINARY_API_SECRET` | `services/cloudinary.py` | Cloudinary API Secret | Fallback to local disk storage |
| `MAX_UPLOAD_SIZE_MB` | `app/main.py` | Maximum upload size validation (20MB) | Defaults to 20 |
| `TESTING` / `PYTEST_CURRENT_TEST`| Multiple modules | Bypasses external calls during testing | Standard production behavior |

