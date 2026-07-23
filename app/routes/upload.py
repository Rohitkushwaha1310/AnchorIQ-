"""
Module 1 — Upload
Accepts a CSV (Excel support can be added the same way with pd.read_excel),
stores it, and returns a file_id the frontend/other endpoints reference.
"""
import uuid
import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.inspection import analyze_dataset

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv",)):
        raise HTTPException(status_code=400, detail="Only CSV files are supported right now.")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Validate it's actually readable as a dataframe before we accept it
    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    profile = analyze_dataset(df)

    return {
        "file_id": file_id,
        "original_filename": file.filename,
        "profile": profile,
    }
