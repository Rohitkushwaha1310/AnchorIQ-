import sys
import os
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import uuid
import shutil

from services.inspection import inspect_dataset
from services.cleaning   import auto_clean
from services.eda        import auto_eda
from services.modeling   import auto_model
from services.insights   import generate_insights


app = FastAPI(
    title = "AnchorIQ",
    description = "Autonomous Data Analysis Platform",
    version = "1.0.0"
)

app.add_middleware(CORSMiddleware,
        allow_origins = ["*"],
        allow_methods = ['*'],
        allow_headers=["*"])

os.makedirs("charts", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/charts", StaticFiles(directory="charts"),
          name="charts")

@app.get("/")
def root():
    return{
        "message": "AnchorIQ API Running!",
        "version": "1.0.0",
    }

@app.get("/health")
def health():
    return{"status": "Healthy"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload ANY CSV → get full analysis!
    Steps: Inspect → Clean → EDA → ML → AI Insights
    """
    print(f"\n{'='*50}")
    print(f" File received: {file.filename}")
    print(f"{'='*50}")

    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Could not read CSV: {str(e)}"}

    print(f" Loaded: {df_raw.shape}")

    #step 1 inspect
    print("\n Step 1: Inspecting dataset...")
    inspection = inspect_dataset(df_raw)
    target     = inspection['target_column']
    print(f"   Target detected: {target}")

    #step 2
    print("\n Step 2: Cleaning  the data...")
    df_clean, clean_report = auto_clean(df_raw.copy())
    print(f"   Clean shape: {df_clean.shape}")


    # STEP 3: EDA 
    print("\n📊 Step 3: Generating EDA charts...")
    # Unique folder per analysis
    session_id = str(uuid.uuid4())[:8]
    chart_dir  = f"charts/{session_id}"
    charts     = auto_eda(df_clean,
                          target=target,
                          save_dir=chart_dir)
    # Convert to URLs
    chart_urls = [
        f"http://localhost:8000/{c}" for c in charts]

    #  STEP 4: ML MODEL 
    print("\n🤖 Step 4: Training ML model...")
    model_results = {}
    if target:
        try:
            model_results = auto_model(
                df_clean.copy(), target)
        except Exception as e:
            print(f"⚠️ ML failed: {e}")
            model_results = {"error": str(e)}

    # STEP 5: ai insights
    print("\n💡 Step 5: Generating AI insights...")
    insights = generate_insights(
        inspection, model_results, target or "unknown")

    print(f"\n✅ Analysis Complete!")
    print(f"{'='*50}\n")

    return {
        "filename"     : file.filename,
        "session_id"   : session_id,
        "inspection"   : inspection,
        "cleaning"     : clean_report,
        "chart_urls"   : chart_urls,
        "model_results": model_results,
        "insights"     : insights,
        "status"       : "✅ Analysis Complete!"
    }



