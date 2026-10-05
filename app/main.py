import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import re

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from services.loader import SUPPORTED_EXTENSIONS, load_any
from services.modeling import predict_new
from services.pdf_report import generate_pdf_report
from services.pipeline import run_analysis

app = FastAPI(title="AnchorIQ API", version="3.0.0",
              description="Autonomous data analysis: upload any data -> insights, predictions, recommendations")

# Set ALLOWED_ORIGINS="https://yourapp.com,http://localhost:8501" in .env for production
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
origins = [o.strip() for o in raw_origins.split(",") if o.strip() and o.strip() != "*"]
if not origins:
    origins = ["http://localhost:8501", "http://127.0.0.1:8501"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

for d in ("charts", "uploads", "reports", "models"):
    os.makedirs(d, exist_ok=True)
app.mount("/charts", StaticFiles(directory="charts"), name="charts")

SESSION_RE = re.compile(r"^[a-f0-9]{8}$")


def _sanitize_filename(name: str | None) -> str:
    raw = os.path.basename(name or "data")
    sanitized = re.sub(r"[^\w\.\-]", "_", raw)
    return sanitized or "data"


@app.get("/")
def root():
    return {"message": "AnchorIQ API v3.0", "supported_formats": list(SUPPORTED_EXTENSIONS),
            "endpoints": ["/analyze", "/generate_report", "/predict/{session_id}",
                          "/download/{session_id}", "/health"]}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "3.0.0"}


async def _run(request: Request, file: UploadFile, analysis_type: str, target_column: str | None):
    content = await file.read()
    safe_name = _sanitize_filename(file.filename)
    try:
        df, meta = load_any(safe_name, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")

    try:
        result = await run_in_threadpool(run_analysis, df, safe_name, analysis_type,
                                         target_column or None, None, meta.get("notes", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    base = str(request.base_url).rstrip("/")
    result["chart_urls"] = [f"{base}/{p}" for p in result.get("chart_paths", [])]
    return result


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...),
                  analysis_type: str = Form("auto"), target_column: str | None = Form(None)):
    """Upload ANY supported file -> full analysis as JSON."""
    return await _run(request, file, analysis_type, target_column)


@app.post("/generate_report")
async def generate_report(request: Request, file: UploadFile = File(...),
                          analysis_type: str = Form("auto"), target_column: str | None = Form(None)):
    result = await _run(request, file, analysis_type, target_column)
    pdf_path = f"reports/AnchorIQ_{analysis_type}_{result['session_id']}.pdf"
    try:
        await run_in_threadpool(generate_pdf_report, result, result.get("chart_paths", []), pdf_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"PDF generation failed: {e}"})
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"AnchorIQ_{analysis_type}_Report.pdf")


@app.get("/download/{session_id}")
def download_enriched(session_id: str):
    """CSV of your data + predictions / risk tiers / segments / anomaly flags."""
    if not SESSION_RE.match(session_id):
        raise HTTPException(400, "Invalid session id")
    path = f"reports/{session_id}_enriched.csv"
    if not os.path.exists(path):
        raise HTTPException(404, "Unknown session")
    return FileResponse(path, media_type="text/csv", filename=f"AnchorIQ_{session_id}_results.csv")


@app.post("/predict/{session_id}")
async def predict(session_id: str, file: UploadFile = File(...)):
    """Score NEW data (same columns as the original) with the model trained in a previous session."""
    if not SESSION_RE.match(session_id):
        raise HTTPException(400, "Invalid session id")
    model_path = f"models/{session_id}.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(404, "No trained model for this session (it needs a target column).")
    content = await file.read()
    safe_name = _sanitize_filename(file.filename)
    try:
        df, _ = load_any(safe_name, content)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to read file for prediction: {e}")
    try:
        scored = await run_in_threadpool(predict_new, model_path, df)
    except Exception as e:
        raise HTTPException(500, detail=f"Prediction scoring failed: {e}")
    out_path = f"reports/{session_id}_new_predictions.csv"
    scored.to_csv(out_path, index=False)
    return FileResponse(out_path, media_type="text/csv", filename="AnchorIQ_new_predictions.csv")
