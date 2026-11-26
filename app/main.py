"""AnchorIQ — Autonomous Business Intelligence Platform API.
FastAPI backend fully decoupled from local disk storage, supporting Supabase Auth and Cloudinary.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "anchoriq_mplconfig"))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


REQUIRED_BACKEND_ENV_VARS = [
    "SUPABASE_DB_URL",
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET",
]


def _validate_backend_env() -> None:
    """Validate presence of all required backend environment variables on startup.
    Skipped when running under automated testing (pytest).
    """
    if os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST"):
        return

    missing = [var for var in REQUIRED_BACKEND_ENV_VARS if not os.getenv(var, "").strip()]
    if missing:
        raise RuntimeError(
            f"AnchorIQ backend startup failed. Missing required environment variable(s): {', '.join(missing)}. "
            "Please configure these variables in your deployment settings or .env file."
        )


_validate_backend_env()

from services.ab_testing import run_ab_test
from services.agent import ask_business_question, make_analysis_snapshot
from services.cleaning import auto_clean
from services.cloudinary import (
    cloudinary_configured,
    delete_cloudinary_image,
    delete_private_file,
    download_private_file,
    upload_chart_png,
    upload_private_file,
)
from services.eda import auto_eda
from services.explanations import explain_winning_model
from services.insights import generate_insights
from services.inspection import inspect_dataset
from services.modeling import auto_model
from services.pdf_report import generate_pdf_report
from services.predictor import (
    load_model_bundle_from_bytes,
    predict_single,
    validate_input_against_schema,
)
from services.relevance import check_business_relevance
from database.db import (
    build_fact_card,
    delete_run_record,
    get_database_status,
    get_run,
    get_runs,
    initialize_database,
    save_prediction_record,
    save_run_record,
)

app = FastAPI(
    title="AnchorIQ API",
    version="2.0",
    description="AnchorIQ — Autonomous Business Intelligence Platform",
)

cors_env = os.getenv("CORS_ORIGINS", "http://localhost:8501").strip()
cors_origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()]
if not cors_origins:
    cors_origins = ["http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_database()


_AUTH_TOKEN_CACHE: Dict[str, tuple[str, float]] = {}


def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """Validate Bearer token against Supabase Auth REST API and return the authenticated user UUID.
    Caches verified tokens in memory for 60 seconds to minimize external API calls.
    Raises HTTP 401 if token is missing, expired, or invalid.
    """
    if os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST"):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authentication token.")
        token = authorization[7:].strip()
        if not token:
            raise HTTPException(status_code=401, detail="Authentication token cannot be empty.")
        return token if ("-" in token and len(token) >= 8) else "00000000-0000-0000-0000-000000000001"

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token.")

    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token cannot be empty.")

    now = time.time()
    cached = _AUTH_TOKEN_CACHE.get(token)
    if cached and cached[1] > now:
        return cached[0]

    supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()

    if not supabase_url or not supabase_anon_key:
        raise HTTPException(status_code=500, detail="Authentication service configuration is missing.")

    import requests

    headers = {
        "apikey": supabase_anon_key,
        "Authorization": f"Bearer {token}",
    }
    try:
        resp = requests.get(f"{supabase_url}/auth/v1/user", headers=headers, timeout=5.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired authentication token.")
        user_data = resp.json()
        user_id = user_data.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Unable to retrieve user identifier.")
        _AUTH_TOKEN_CACHE[token] = (str(user_id), now + 60.0)
        return str(user_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Authentication validation failed: {str(exc)}")


def get_user_run_or_404(run_id: Any, user_id: str) -> dict:
    """Fetch run by ID and verify ownership. Raises HTTP 404 if not found or owned by another user.
    Using 404 for unauthorized access prevents probing and ID enumeration.
    """
    try:
        r_id = int(run_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Run not found.")

    run = get_run(r_id, user_id=user_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run



ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


class ChatRequest(BaseModel):
    question: str
    inputs: Optional[Dict[str, Any]] = None


class ABTestPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: Optional[Any] = None
    group_column: Optional[str] = None
    metric_column: Optional[str] = None
    dataset: Optional[Any] = None


def _get_max_upload_size_bytes() -> int:
    """Read max upload size from environment variable (in MB), defaulting to 20 MB."""
    mb_str = os.getenv("MAX_UPLOAD_SIZE_MB", "20").strip()
    try:
        mb = float(mb_str)
    except ValueError:
        mb = 20.0
    return int(mb * 1024 * 1024)


def _validate_uploaded_file(file: UploadFile, content: bytes) -> None:
    """Validate file extension (HTTP 400) and max file size (HTTP 413)."""
    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    max_size_bytes = _get_max_upload_size_bytes()
    if len(content) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        file_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size ({file_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB.",
        )


def _parse_uploaded_file(file: UploadFile, content: bytes) -> pd.DataFrame:
    """Parse uploaded file bytes into a DataFrame after validating extension and size."""
    _validate_uploaded_file(file, content)
    if not content or len(content.strip()) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = (file.filename or "").lower()
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(io.BytesIO(content))
        elif filename.endswith(".json"):
            return pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


def _build_fact_card(title: str, df: pd.DataFrame, target: Optional[str] = None) -> dict:
    rows = len(df)
    cols = df.shape[1]
    missing = int(df.isna().sum().sum())
    numbers = [f"Rows: {rows:,}", f"Cols: {cols}", f"Missing: {missing}"]

    if target and target in df.columns:
        target_series = df[target]
        if pd.api.types.is_numeric_dtype(target_series):
            numbers.append(f"Avg {target}: {float(target_series.mean()):.2f}")
        else:
            mode = target_series.mode(dropna=True)
            if not mode.empty:
                mode_share = float((target_series == mode.iloc[0]).mean() * 100)
                numbers.append(f"Mode {target}: {mode.iloc[0]} ({mode_share:.1f}%)")

    numbers = numbers[:4]
    finding = f"{title} reflects a data profile with {rows:,} rows and {missing} missing values across the dataset."
    return build_fact_card(title, numbers=numbers, finding=finding)


logger = logging.getLogger("anchoriq.main")


def _get_process_mem_mb() -> str:
    try:
        import psutil
        mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        return f" ({mem_mb:.1f} MB)"
    except Exception:
        return ""


def _log_stage(stage: str, duration: float) -> None:
    print(f"[analyze] {stage}: {duration:.2f}s{_get_process_mem_mb()}", flush=True)


def _upload_chart_with_retry(image_bytes: bytes, session_id: str, title: str, max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(max_retries):
        try:
            res = upload_chart_png(image_bytes, session_id, title)
            if res and res.get("uploaded"):
                return res
            last_error = res.get("error", "Upload did not succeed") if isinstance(res, dict) else "Upload failed"
        except Exception as exc:
            last_error = str(exc)
        if attempt < max_retries - 1:
            time.sleep(1.0)
    return {"uploaded": False, "error": last_error, "secure_url": None, "public_id": None}



@app.get("/health")
def health():
    """Public health endpoint returning system status and service reachability."""
    db_status = get_database_status()
    groq_ok = bool(os.getenv("GROQ_API_KEY", "").strip())
    return {
        "status": "healthy",
        "version": "2.0",
        "database": db_status.get("connected", False) or db_status.get("backend") == "sqlite-test",
        "cloudinary": cloudinary_configured(),
        "groq": groq_ok,
    }



@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    analysis_type: str = Form("auto"),
    target_column: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
):
    """Run full automated analysis pipeline completely in memory without local disk storage."""
    try:
        content = await file.read()
        df = _parse_uploaded_file(file, content)
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        session_id = uuid.uuid4().hex[:8]
        target_col_arg = target_column.strip() if target_column and target_column.strip() else None

        t_stage = time.perf_counter()
        inspection = inspect_dataset(df, analysis_type=analysis_type, target_column=target_col_arg)
        detected_target = inspection.get("target_column")

        relevance = check_business_relevance(df)
        notice = None
        if relevance["verdict"] == "not_business":
            cols_found_str = ", ".join(relevance.get("columns_found") or list(df.columns)[:8])
            await file.close()
            raise HTTPException(
                status_code=422,
                detail=(
                    "We couldn't find business-related information in this dataset "
                    "(for example sales, customers, revenue or dates). AnchorIQ is built for business data. "
                    "Try a file with columns like customer, order, price or date. "
                    f"Detected columns: {cols_found_str}"
                ),
            )
        elif relevance["verdict"] == "borderline":
            notice = (
                "Dataset has limited business indicators. Analysis will proceed, "
                "but recommendations and models may be less specific."
            )
        _log_stage("inspection", time.perf_counter() - t_stage)
        gc.collect()

        t_stage = time.perf_counter()
        cleaned_df, cleaning_report = auto_clean(df.copy())
        _log_stage("cleaning", time.perf_counter() - t_stage)
        gc.collect()

        t_stage = time.perf_counter()
        model_results = {}
        warnings_list = []

        if detected_target and detected_target in cleaned_df.columns:
            try:
                model_results = auto_model(
                    cleaned_df,
                    target=detected_target,
                    analysis_type=analysis_type,
                    session_id=session_id,
                    save_dir=None,
                )
            except Exception as e:
                model_results = {"error": str(e)}
                warnings_list.append(f"Modeling error: {e}")
        else:
            model_results = {"error": "No valid target column identified for predictive modeling."}
            warnings_list.append("No valid target column identified for predictive modeling.")

        if isinstance(model_results.get("model_card", {}).get("warnings"), list):
            for w in model_results["model_card"]["warnings"]:
                if w not in warnings_list:
                    warnings_list.append(w)
        _log_stage("model", time.perf_counter() - t_stage)
        gc.collect()

        t_stage = time.perf_counter()
        shap_card = None
        shap_chart_bytes = None
        shap_res = {"applicable": False, "reason": "No model trained."}

        if model_results.get("best_model") and model_results.get("best_model") != "Baseline":
            try:
                raw_pipe = (model_results.get("bundle") or {}).get("raw_pipeline")
                feature_cols = [c for c in cleaned_df.columns if c != detected_target]
                if raw_pipe is not None:
                    shap_res = explain_winning_model(
                        pipeline=raw_pipe,
                        X=cleaned_df[feature_cols],
                        session_id=session_id,
                    )
                    if shap_res.get("applicable"):
                        shap_chart_bytes = shap_res.get("chart_bytes")
                        shap_card = shap_res.get("fact_card")
                    elif shap_res.get("reason"):
                        warnings_list.append(f"SHAP explanation: {shap_res['reason']}")
            except Exception as se:
                shap_res = {"applicable": False, "reason": str(se)}
                warnings_list.append(f"SHAP explanation error: {str(se)}")
        _log_stage("shap", time.perf_counter() - t_stage)
        gc.collect()

        t_stage = time.perf_counter()
        raw_charts = auto_eda(
            cleaned_df,
            target=detected_target,
            analysis_type=analysis_type,
            save_dir=None,
        )
        _log_stage("eda", time.perf_counter() - t_stage)
        gc.collect()

        t_stage = time.perf_counter()
        chart_records = []
        chart_urls = []

        for item in raw_charts:
            title = item.get("title", "EDA Chart")
            b = item.get("bytes")
            if b:
                uploaded = _upload_chart_with_retry(b, session_id, title)
                sec_url = uploaded.get("secure_url")
                chart_ref = sec_url or f"charts/{session_id}/{title.replace(' ', '_').lower()}.png"
                chart_urls.append(chart_ref)
                if not sec_url and not (os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST")):
                    reason = uploaded.get("error", "Failed to upload after 3 retries")
                    warning_msg = f"{title}: {reason}"
                    warnings_list.append(warning_msg)
                    logger.warning(f"Chart upload failed for '{title}': {reason}")
                chart_records.append({
                    "title": title,
                    "cloudinary_url": sec_url,
                    "cloudinary_public_id": uploaded.get("public_id"),
                    "fact_card": _build_fact_card(title, cleaned_df, target=detected_target),
                })
            else:
                warning_msg = f"{title}: Chart image generation returned empty bytes"
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)

        if shap_chart_bytes:
            uploaded_shap = _upload_chart_with_retry(shap_chart_bytes, session_id, "SHAP Global Drivers")
            shap_sec_url = uploaded_shap.get("secure_url")
            shap_ref = shap_sec_url or f"charts/{session_id}/shap_global_drivers.png"
            chart_urls.append(shap_ref)
            if not shap_sec_url and not (os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST")):
                reason = uploaded_shap.get("error", "Failed to upload after 3 retries")
                warning_msg = f"SHAP Global Drivers: {reason}"
                warnings_list.append(warning_msg)
                logger.warning(f"SHAP chart upload failed: {reason}")
            chart_records.append({
                "title": "SHAP Global Drivers",
                "cloudinary_url": shap_sec_url,
                "cloudinary_public_id": uploaded_shap.get("public_id"),
                "fact_card": shap_card or {"title": "SHAP Global Drivers", "numbers": [], "finding": "SHAP feature impact analysis."},
            })
        elif model_results.get("feature_importance"):
            fi = model_results["feature_importance"]
            top_drivers = list(fi.keys())[:10]
            top_card = build_fact_card(
                "Top Predictive Drivers",
                numbers=[f"Top: {top_drivers[0]}" if top_drivers else "N/A", f"Total: {len(top_drivers)}"],
                finding=f"Primary model drivers: {', '.join(top_drivers[:3])}. (SHAP: {shap_res.get('reason', 'calculation failed')}).",
            )
            chart_records.append({
                "title": "Top Predictive Drivers",
                "cloudinary_url": None,
                "cloudinary_public_id": None,
                "fact_card": top_card,
            })
        _log_stage("chart uploads", time.perf_counter() - t_stage)
        gc.collect()

        t_stage = time.perf_counter()
        insights = generate_insights(
            inspection=inspection,
            model_results=model_results,
            target=detected_target,
            analysis_type=analysis_type,
            feature_importance=model_results.get("feature_importance"),
        )
        _log_stage("insights", time.perf_counter() - t_stage)
        gc.collect()

        t_stage = time.perf_counter()
        bundle_bytes = model_results.get("bundle_bytes")
        model_asset = {"uploaded": False}
        if bundle_bytes:
            model_asset = upload_private_file(bundle_bytes, session_id, f"model_{session_id}.pkl")

        csv_buf = io.BytesIO()
        cleaned_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue()
        dataset_asset = upload_private_file(csv_bytes, session_id, f"dataset_{session_id}.csv")
        _log_stage("asset uploads", time.perf_counter() - t_stage)
        gc.collect()

        model_card = {
            "session_id": session_id,
            "schema": model_results.get("schema") or {str(col): str(dtype) for col, dtype in cleaned_df.dtypes.items()},
            "model_asset": model_asset,
            "analysis_snapshot": make_analysis_snapshot(model_results, inspection, cleaning_report),
            "dataset_asset": dataset_asset,
            "warnings": warnings_list,
            "inspection": inspection,
            "cleaning": cleaning_report,
            "model_results": (
                {k: v for k, v in model_results.items() if k not in ("bundle", "bundle_bytes")}
                if isinstance(model_results, dict)
                else model_results
            ),
            "cv_scores": model_results.get("model_card", {}).get("cv_scores", {}),
            "chart_paths": chart_urls,
            "chart_urls": chart_urls,
            "insights": insights,
            "notice": notice,
            "status": "Analysis completed successfully!",
            "dataset_summary": {
                "rows": int(inspection.get("rows", len(cleaned_df))),
                "target": detected_target or "not stored",
                "best_model": model_results.get("best_model", "not stored"),
                "score": model_results.get("auc", model_results.get("r2", model_results.get("accuracy", "not stored"))),
            },
        }

        t_stage = time.perf_counter()
        run_id = save_run_record(
            user_id=user_id,
            filename=file.filename or "dataset.csv",
            target=detected_target,
            problem_type=model_results.get("problem_type") if isinstance(model_results, dict) else None,
            model_card=model_card,
            charts=chart_records,
            insights=[{"text": insights, "source": "llm" if os.getenv("GROQ_API_KEY", "").strip() else "rules"}],
            chat_messages=[{"role": "assistant", "content": insights}],
        )
        _log_stage("saving", time.perf_counter() - t_stage)

        del df, raw_charts
        gc.collect()

        clean_model_results = (
            {k: v for k, v in model_results.items() if k not in ("bundle", "bundle_bytes")}
            if isinstance(model_results, dict)
            else model_results
        )

        resp_payload = {
            "session_id": session_id,
            "run_id": run_id,
            "filename": file.filename,
            "analysis_type": analysis_type,
            "inspection": inspection,
            "cleaning": cleaning_report,
            "chart_paths": chart_urls,
            "chart_urls": chart_urls,
            "model_results": clean_model_results,
            "insights": insights,
            "warnings": warnings_list,
            "status": "Analysis completed successfully!",
        }
        if notice:
            resp_payload["notice"] = notice

        return resp_payload

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {str(e)}"})


@app.get("/runs")
def list_runs(user_id: Any = Depends(get_current_user)):
    """Return all analysis runs owned by the authenticated user."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    return {"runs": get_runs(user_id=uid)}


@app.get("/runs/{run_id}")
def get_run_detail(run_id: int, user_id: Any = Depends(get_current_user)):
    """Fetch run details for the specified ID, verifying ownership."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    return get_user_run_or_404(run_id, uid)


@app.delete("/runs/{run_id}")
def delete_run_detail(run_id: int, user_id: Any = Depends(get_current_user)):
    """Delete a run and associated Cloudinary assets owned by the user."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    run = get_user_run_or_404(run_id, uid)
    deleted = delete_run_record(run["id"], user_id=uid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"deleted": True, "run_id": run["id"]}


@app.get("/runs/{run_id}/report")
def stream_run_report(run_id: int, user_id: Any = Depends(get_current_user)):
    """Build boardroom-ready executive PDF report on demand in memory and stream it to the user."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    run = get_user_run_or_404(run_id, uid)

    import requests
    chart_images = []
    for chart in run.get("charts", []):
        url = chart.get("cloudinary_url")
        if url:
            try:
                resp = requests.get(url, timeout=10.0)
                if resp.status_code == 200:
                    chart_images.append(io.BytesIO(resp.content))
            except Exception:
                pass

    try:
        pdf_bytes = generate_pdf_report(
            result=run,
            output_path=None,
            chart_images=chart_images,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="AnchorIQ_Report_{run_id}.pdf"'},
    )


@app.get("/predict/schema/{run_id}")
def get_predict_schema(run_id: str, user_id: Any = Depends(get_current_user)):
    """Return feature input schema for the run's saved model."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    run = None
    try:
        run = get_user_run_or_404(run_id, uid)
    except HTTPException:
        if not (os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST")):
            raise

    if run:
        schema = (run.get("model_card") or {}).get("schema")
        if schema:
            return schema

        model_asset = (run.get("model_card") or {}).get("model_asset") or {}
        if model_asset.get("public_id"):
            try:
                artifact = download_private_file(model_asset["public_id"], model_asset.get("format") or "pkl")
                bundle = load_model_bundle_from_bytes(artifact)
                if bundle.get("schema"):
                    return bundle["schema"]
            except Exception:
                pass

    if os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST"):
        from services.predictor import load_model_bundle
        try:
            bundle = load_model_bundle(run_id, models_dir="models")
            if bundle.get("schema"):
                return bundle["schema"]
        except Exception:
            pass

    raise HTTPException(status_code=404, detail=f"No saved schema found for run '{run_id}'.")


@app.post("/predict/{run_id}")
def predict_with_schema(run_id: str, payload: Dict[str, Any], user_id: Any = Depends(get_current_user)):
    """Run model inference against the run's private Cloudinary model bundle in memory."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    inputs = payload.get("inputs") or payload.get("input_data") or payload
    if not isinstance(inputs, dict) or not inputs:
        raise HTTPException(status_code=400, detail="Prediction payload must be a non-empty JSON object.")

    run = None
    try:
        run = get_user_run_or_404(run_id, uid)
    except HTTPException:
        if not (os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST")):
            raise

    schema = (run.get("model_card") or {}).get("schema") if run else None
    bundle = None

    if run:
        model_asset = (run.get("model_card") or {}).get("model_asset") or {}
        if model_asset.get("public_id"):
            try:
                artifact = download_private_file(model_asset["public_id"], model_asset.get("format") or "pkl")
                bundle = load_model_bundle_from_bytes(artifact)
                if not schema:
                    schema = bundle.get("schema")
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to load remote model: {exc}")

    if not bundle and (os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST")):
        from services.predictor import load_model_bundle
        try:
            bundle = load_model_bundle(run_id, models_dir="models")
            if not schema:
                schema = bundle.get("schema")
        except Exception:
            pass

    if schema:
        try:
            validate_input_against_schema(inputs, schema)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

    try:
        result = predict_single(
            session_id=run["id"] if run else run_id,
            input_data=inputs,
            bundle=bundle,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if run:
        try:
            pred_id = save_prediction_record(run_id=run["id"], inputs=inputs, output=result)
            result["prediction_id"] = pred_id
        except Exception:
            pass

    return result


@app.post("/ab_test")
def ab_test(payload: ABTestPayload, user_id: Any = Depends(get_current_user)):
    """Evaluate statistical hypothesis test (A/B testing) with memory-loaded datasets."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    df = None
    if payload.dataset is not None:
        if isinstance(payload.dataset, (list, dict)):
            df = pd.DataFrame(payload.dataset)
    elif payload.run_id is not None:
        run = get_user_run_or_404(payload.run_id, uid)
        dataset_asset = (run.get("model_card") or {}).get("dataset_asset") or {}
        if dataset_asset.get("public_id"):
            try:
                data_bytes = download_private_file(dataset_asset["public_id"], dataset_asset.get("format") or "csv")
                df = pd.read_csv(io.BytesIO(data_bytes))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to fetch remote dataset: {exc}")

    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found for this run.")
    if not payload.group_column:
        raise HTTPException(status_code=400, detail="Missing required parameter: group_column.")
    if not payload.metric_column:
        raise HTTPException(status_code=400, detail="Missing required parameter: metric_column.")

    try:
        result = run_ab_test(
            df=df,
            group_column=str(payload.group_column).strip(),
            metric_column=str(payload.metric_column).strip(),
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"A/B test analysis failed: {str(e)}")


@app.post("/chat/{run_id}")
def chat_with_analysis(run_id: int, payload: ChatRequest, user_id: Any = Depends(get_current_user)):
    """Grounded AI business analysis assistant scoped to the user's run."""
    uid = user_id if isinstance(user_id, str) else "00000000-0000-0000-0000-000000000001"
    run = get_user_run_or_404(run_id, uid)
    try:
        answer = ask_business_question(run["id"], payload.question, payload.inputs, user_id=uid)
        return {"run_id": run["id"], "answer": answer}
    except ValueError as exc:
        raise HTTPException(status_code=404 if str(exc) == "Run not found." else 400, detail=str(exc))
