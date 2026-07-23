# InsightAI — Autonomous Data Analyst

## Status: Phase 1 in progress (Module 1: Upload, Module 2: Automatic Inspection)

## Run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Then test the upload endpoint:

```bash
curl -X POST -F "file=@sample_churn.csv" http://localhost:8000/api/upload
```

Or open http://localhost:8000/docs for the interactive Swagger UI.

## What's built so far

- `app/main.py` — FastAPI entrypoint
- `app/routes/upload.py` — Module 1: CSV upload + validation, returns `file_id` + auto profile
- `app/services/inspection.py` — Module 2: `analyze_dataset(df)` — rows, cols, missing, duplicates,
  numerical/categorical/datetime column detection, target column guess

## Next up

- Module 3: `clean_data(df)` — automatic cleaning
- Module 5/6: automatic EDA + chart generation
- Streamlit frontend to visualize this live
