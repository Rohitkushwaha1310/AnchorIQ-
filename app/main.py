import os
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mplconfig")))
import sys
import uuid
import re
import io
import json
import base64
import pandas as pd
from typing import Optional, Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from services.inspection import inspect_dataset
from services.relevance import check_business_relevance
from services.cleaning import auto_clean
from services.eda import auto_eda
from services.modeling import auto_model
from services.insights import generate_insights
from services.pdf_report import generate_pdf_report
from services.ab_testing import run_ab_test
from services.predictor import (
    predict_single,
    validate_input_against_schema,
    load_model_bundle,
    RISK_LOW_THRESHOLD,
    RISK_HIGH_THRESHOLD,
)
from services.explanations import explain_winning_model
from services.cloudinary import upload_chart_png
from database.db import (
    initialize_database,
    save_run_record,
    get_runs,
    get_run,
    delete_run_record,
    build_fact_card,
    save_prediction_record,
)
from services.agent import ask_business_question

app = FastAPI(
    title="AnchorIQ API",
    version="2.0",
    description="AnchorIQ — Autonomous Business Intelligence Platform"
)

# CORS: Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist
for folder in ["charts", "models", "reports", "uploads"]:
    os.makedirs(folder, exist_ok=True)

initialize_database()

# Static file mount: /charts -> charts/ directory
app.mount("/charts", StaticFiles(directory="charts"), name="charts")


class PredictRequest(BaseModel):
    session_id: str
    input_data: dict


class ChatRequest(BaseModel):
    question: str
    inputs: Optional[Dict[str, Any]] = None


ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


def _get_max_upload_size_bytes() -> int:
    """Read max upload size from environment variable (in MB), defaulting to 20 MB."""
    mb_str = os.getenv("MAX_UPLOAD_SIZE_MB", "20").strip()
    try:
        mb = float(mb_str)
    except ValueError:
        mb = 20.0
    return int(mb * 1024 * 1024)


def _validate_uploaded_file(file: UploadFile, content: bytes) -> None:
    """Validate allowed extension (HTTP 400) and max file size (HTTP 413)."""
    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    max_size_bytes = _get_max_upload_size_bytes()
    if len(content) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        file_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size ({file_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
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


def _build_fact_card(title: str, df: pd.DataFrame, target: Optional[str] = None, local_path: Optional[str] = None) -> dict:
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
    return build_fact_card(title, local_path=local_path, numbers=numbers, finding=finding)


def _build_model_card(df: pd.DataFrame, model_path: Optional[str] = None, detailed_schema: Optional[dict] = None) -> dict:
    if detailed_schema:
        schema = detailed_schema
    else:
        schema = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    model_card = {"schema": schema}
    if model_path and os.path.exists(model_path):
        with open(model_path, "rb") as fh:
            model_card["model_file_b64"] = base64.b64encode(fh.read()).decode("utf-8")
            model_card["saved_path"] = model_path
    return model_card


@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0"}


@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    analysis_type: str = Form("auto"),
    target_column: Optional[str] = Form(None)
):
    try:
        content = await file.read()
        df = _parse_uploaded_file(file, content)
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        session_id = uuid.uuid4().hex[:8]
        charts_dir = os.path.join("charts", session_id)
        os.makedirs(charts_dir, exist_ok=True)

        target_col_arg = target_column.strip() if target_column and target_column.strip() else None

        # 1. Inspection
        inspection = inspect_dataset(df, analysis_type=analysis_type, target_column=target_col_arg)
        detected_target = inspection.get("target_column")

        # Business Relevance Gate
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
                )
            )
        elif relevance["verdict"] == "borderline":
            notice = (
                "Dataset has limited business indicators. Analysis will proceed, "
                "but recommendations and models may be less specific."
            )

        # 2. Cleaning
        cleaned_df, cleaning_report = auto_clean(df.copy())

        # 3. ML Modeling (if target available)
        model_results = {}
        shap_card = None
        shap_res = {"applicable": False, "reason": "No model trained."}
        if detected_target and detected_target in cleaned_df.columns:
            try:
                model_results = auto_model(
                    cleaned_df,
                    target=detected_target,
                    analysis_type=analysis_type,
                    save_dir="models",
                    session_id=session_id
                )
                if model_results.get("best_model") and model_results.get("best_model") != "Baseline":
                    try:
                        raw_bundle = load_model_bundle(session_id, models_dir="models")
                        best_pipe = raw_bundle.get("pipeline")
                        feature_cols = [c for c in cleaned_df.columns if c != detected_target]
                        shap_res = explain_winning_model(
                            pipeline=best_pipe,
                            X=cleaned_df[feature_cols],
                            session_id=session_id,
                            charts_dir=charts_dir
                        )
                        if shap_res.get("applicable") and shap_res.get("chart_path"):
                            chart_paths.append(shap_res["chart_path"])
                        if shap_res.get("fact_card"):
                            shap_card = shap_res["fact_card"]
                    except Exception as se:
                        shap_res = {"applicable": False, "reason": str(se)}
            except Exception as e:
                model_results = {"error": str(e)}
        else:
            model_results = {"error": "No valid target column identified for predictive modeling."}

        # 4. EDA Charts (8 charts)
        chart_paths = []
        try:
            chart_paths = auto_eda(
                cleaned_df,
                target=detected_target,
                analysis_type=analysis_type,
                save_dir=charts_dir,
            )
        except Exception as e:
            print(f"[WARN] EDA failed: {e}")

        # Construct URLs for charts
        base_url = str(request.base_url).rstrip("/")
        chart_urls = [f"{base_url}/{p.replace(os.sep, '/')}" for p in chart_paths]

        # 5. AI Insights
        insights = generate_insights(
            inspection=inspection,
            model_results=model_results,
            target=detected_target,
            analysis_type=analysis_type,
            feature_importance=model_results.get("feature_importance")
        )

        chart_titles = [
            "Feature Distributions",
            "Correlation Heatmap",
            "Target Distribution",
            "Categorical Impact",
            "Numerical Boxplots",
            "Domain Analysis",
        ]
        chart_records = []
        for index, chart_path in enumerate(chart_paths):
            title = chart_titles[index] if index < len(chart_titles) else os.path.basename(chart_path)
            local_url = f"{base_url}/{chart_path.replace(os.sep, '/')}"
            uploaded = upload_chart_png(chart_path, str(session_id), title)
            secure_url = uploaded.get("secure_url") or local_url
            public_id = uploaded.get("public_id")
            chart_records.append({
                "title": title,
                "cloudinary_url": secure_url,
                "cloudinary_public_id": public_id,
                "fact_card": _build_fact_card(title, cleaned_df, target=detected_target, local_path=chart_path),
            })

        if shap_card:
            chart_records.append({
                "title": shap_card.get("title", "SHAP Global Drivers"),
                "cloudinary_url": None,
                "cloudinary_public_id": None,
                "fact_card": shap_card,
            })
        elif model_results.get("feature_importance"):
            fi = model_results["feature_importance"]
            top_drivers = list(fi.keys())[:10]
            top_card = build_fact_card(
                "Top Predictive Drivers",
                numbers=[f"Top: {top_drivers[0]}" if top_drivers else "N/A", f"Total: {len(top_drivers)}"],
                finding=f"Primary model drivers: {', '.join(top_drivers[:3])}. (SHAP: {shap_res.get('reason', 'library unavailable')})."
            )
            chart_records.append({
                "title": "Top Predictive Drivers",
                "cloudinary_url": None,
                "cloudinary_public_id": None,
                "fact_card": top_card,
            })

        model_path = os.path.join("models", f"{session_id}.pkl")
        if os.path.exists("models/model.pkl"):
            import shutil
            shutil.copy2("models/model.pkl", model_path)

        model_card = _build_model_card(
            cleaned_df,
            model_path if os.path.exists(model_path) else None,
            detailed_schema=model_results.get("schema")
        )
        model_card["dataset_summary"] = {
            "rows": int(inspection.get("rows", len(cleaned_df))),
            "target": detected_target or "not stored",
            "best_model": model_results.get("best_model", "not stored"),
            "score": model_results.get("auc", model_results.get("r2", model_results.get("accuracy", "not stored"))),
        }
        run_id = save_run_record(
            filename=file.filename,
            target=detected_target,
            problem_type=model_results.get("problem_type") if isinstance(model_results, dict) else None,
            model_card=model_card,
            charts=chart_records,
            insights=[{"text": insights, "source": "llm" if os.getenv("GROQ_API_KEY", "").strip() else "rules"}],
            chat_messages=[{"role": "assistant", "content": insights}],
        )

        # Persist run-specific artifacts
        import shutil
        src_model = os.path.join("models", f"{session_id}_model.pkl")
        dst_model = os.path.join("models", f"{run_id}_model.pkl")
        if os.path.exists(src_model):
            shutil.copy2(src_model, dst_model)
        src_schema = os.path.join("models", f"{session_id}_schema.json")
        dst_schema = os.path.join("models", f"{run_id}_schema.json")
        if os.path.exists(src_schema):
            shutil.copy2(src_schema, dst_schema)

        # Save dataset for A/B testing
        cleaned_df.to_csv(os.path.join("uploads", f"{run_id}.csv"), index=False)

        resp_payload = {
            "session_id": session_id,
            "run_id": run_id,
            "filename": file.filename,
            "analysis_type": analysis_type,
            "inspection": inspection,
            "cleaning": cleaning_report,
            "chart_paths": chart_paths,
            "chart_urls": [chart.get("cloudinary_url") for chart in chart_records],
            "model_results": model_results,
            "insights": insights,
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
def list_runs():
    return {"runs": get_runs()}


@app.get("/runs/{run_id}")
def get_run_detail(run_id: int):
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@app.post("/chat/{run_id}")
def chat_with_analysis(run_id: int, payload: ChatRequest):
    try:
        return {"run_id": run_id, "answer": ask_business_question(run_id, payload.question, payload.inputs)}
    except ValueError as exc:
        raise HTTPException(status_code=404 if str(exc) == "Run not found." else 400, detail=str(exc))


@app.delete("/runs/{run_id}")
def delete_run_detail(run_id: int):
    deleted = delete_run_record(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {"deleted": True, "run_id": run_id}


@app.get("/predict/schema/{run_id}")
def get_predict_schema(run_id: str):
    """Return the input fields from the saved schema."""
    schema_path = os.path.join("models", f"{run_id}_schema.json")
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        run_int = int(run_id)
        run = get_run(run_int)
        if run and run.get("model_card", {}).get("schema"):
            return run["model_card"]["schema"]
    except Exception:
        pass

    try:
        bundle = load_model_bundle(run_id, models_dir="models")
        if bundle.get("schema"):
            return bundle["schema"]
    except Exception:
        pass

    raise HTTPException(status_code=404, detail=f"No saved schema found for run '{run_id}'.")


@app.post("/predict/{run_id}")
def predict_with_schema(run_id: str, payload: Dict[str, Any]):
    """Validate against the schema, run the saved pipeline, return prediction, probability, risk and reasons."""
    inputs = payload.get("inputs") or payload.get("input_data") or payload
    if not isinstance(inputs, dict) or not inputs:
        raise HTTPException(status_code=400, detail="Prediction payload must be a non-empty JSON object.")

    # Fetch schema for validation
    schema = None
    schema_path = os.path.join("models", f"{run_id}_schema.json")
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception:
            pass

    if not schema:
        try:
            run_int = int(run_id)
            run = get_run(run_int)
            if run and run.get("model_card"):
                schema = run["model_card"].get("schema")
        except Exception:
            pass

    if not schema:
        try:
            bundle = load_model_bundle(run_id, models_dir="models")
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
            session_id=run_id,
            input_data=inputs,
            models_dir="models"
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Persist to database predictions table
    try:
        run_int = int(run_id)
        pred_id = save_prediction_record(run_id=run_int, inputs=inputs, output=result)
        result["prediction_id"] = pred_id
    except Exception:
        pass

    return result


@app.post("/predict")
async def predict(payload: PredictRequest):
    try:
        result = predict_single(
            session_id=payload.session_id,
            input_data=payload.input_data,
            models_dir="models"
        )
        return result
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class ABTestPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: Optional[Any] = None
    group_column: Optional[str] = None
    metric_column: Optional[str] = None
    dataset: Optional[Any] = None


@app.post("/ab_test")
def ab_test(payload: ABTestPayload):
    """
    POST /ab_test with run_id (or dataset), group_column, metric_column.
    Detect binary vs continuous metric.
    Binary -> chi-square or two-proportion z-test.
    Continuous -> Welch's t-test plus Mann-Whitney.
    More than two groups -> ANOVA/Kruskal-Wallis.
    """
    df = None
    if payload.dataset is not None:
        if isinstance(payload.dataset, list):
            df = pd.DataFrame(payload.dataset)
        elif isinstance(payload.dataset, dict):
            df = pd.DataFrame(payload.dataset)
    elif payload.run_id is not None:
        req_run_id = str(payload.run_id)
        candidates = [
            os.path.join("uploads", f"{req_run_id}.csv"),
            os.path.join("reports", f"{req_run_id}_enriched.csv"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path:
            df = pd.read_csv(path)
        else:
            raise HTTPException(status_code=404, detail=f"No dataset found for run '{req_run_id}'.")

    if df is None:
        raise HTTPException(status_code=400, detail="Must provide either run_id or dataset.")
    if not payload.group_column:
        raise HTTPException(status_code=400, detail="Missing required parameter: group_column.")
    if not payload.metric_column:
        raise HTTPException(status_code=400, detail="Missing required parameter: metric_column.")

    try:
        result = run_ab_test(df=df, group_column=str(payload.group_column).strip(), metric_column=str(payload.metric_column).strip())
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"A/B test analysis failed: {str(e)}")


@app.post("/generate_report")
async def generate_report(
    request: Request,
    file: UploadFile = File(...),
    analysis_type: str = Form("auto"),
    target_column: Optional[str] = Form(None)
):
    try:
        content = await file.read()
        df = _parse_uploaded_file(file, content)
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        session_id = uuid.uuid4().hex[:8]
        charts_dir = os.path.join("charts", session_id)
        os.makedirs(charts_dir, exist_ok=True)

        target_col_arg = target_column.strip() if target_column and target_column.strip() else None

        inspection = inspect_dataset(df, analysis_type=analysis_type, target_column=target_col_arg)
        detected_target = inspection.get("target_column")
        cleaned_df, cleaning_report = auto_clean(df.copy())

        model_results = {}
        if detected_target and detected_target in cleaned_df.columns:
            try:
                model_results = auto_model(
                    cleaned_df,
                    target=detected_target,
                    analysis_type=analysis_type,
                    save_dir="models",
                    session_id=session_id
                )
            except Exception as e:
                model_results = {"error": str(e)}

        chart_paths = auto_eda(
            cleaned_df,
            target=detected_target,
            analysis_type=analysis_type,
            save_dir=charts_dir,
            rf_model=model_results.get("feature_importance")
        )

        insights = generate_insights(
            inspection=inspection,
            model_results=model_results,
            target=detected_target,
            analysis_type=analysis_type,
            feature_importance=model_results.get("feature_importance")
        )

        analysis_payload = {
            "session_id": session_id,
            "filename": file.filename,
            "analysis_type": analysis_type,
            "inspection": inspection,
            "cleaning": cleaning_report,
            "chart_paths": chart_paths,
            "model_results": model_results,
            "insights": insights,
        }

        pdf_path = os.path.join("reports", f"AnchorIQ_{session_id}_report.pdf")
        generate_pdf_report(analysis_payload, chart_paths=chart_paths, output_path=pdf_path)

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF generation failed to output file.")

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"AnchorIQ_{analysis_type}_Report_{session_id}.pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Report generation failed: {str(e)}"})
