# ⚓ AnchorIQ

**AnchorIQ** is an autonomous business intelligence and data analytics platform. It automatically ingests business data (CSV, Excel, JSON), validates and cleans it, performs exploratory data analysis (generating high-resolution charts), benchmarks multiple machine learning algorithms, tunes classification decision thresholds, and generates an executive AI business insights report alongside an exportable executive PDF report.

---

## 🌟 Features

### Built & Functional (v2.0)
- **Automated Inspection (`services/inspection.py`)**: Smart target detection (4 priority levels), column typing, missing values profiling, and duplicate analysis.
- **Auto Data Cleaning (`services/cleaning.py`)**: Deduplication, string whitespace trimming, date/datetime coercion, and null imputation using medians (numeric) or modes (categorical).
- **Automated Exploratory Data Analysis (`services/eda.py`)**: Generates 8 server-safe Matplotlib/Seaborn charts (Distributions, Correlation Matrix, Target Distribution, Categorical vs Target, Boxplots, Outliers, Missing Matrix, and Feature Importance).
- **Multi-Model Machine Learning Engine (`services/modeling.py`)**:
  - Automatically handles both classification and regression.
  - Benchmarks Logistic Regression / Ridge, Random Forest, and XGBoost with standard scaling.
  - Cross-validation with StratifiedKFold / KFold.
  - Automated threshold cost-tuning for classification ($FN = \$500, FP = \$50$).
  - Serializes trained pipelines to `models/{session_id}_model.pkl`.
- **AI Executive Insights (`services/insights.py`)**:
  - Generates strategic business analysis using Groq's free LLM tiers (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`).
  - Seamlessly falls back to a deterministic rule-based business intelligence engine if no API key is provided or offline.
- **Upload Safety & Validation (`app/main.py`)**: Allowed extensions check (`.csv`, `.xlsx`, `.xls`, `.json`) with HTTP 400 rejection and configurable upload size cap (default 20 MB via `MAX_UPLOAD_SIZE_MB`) with HTTP 413 rejection.
- **Automated PDF Report Generation (`services/pdf_report.py`)**: Generates a downloadable C-suite ready PDF report via ReportLab.
- **Manual Prediction Engine (`services/predictor.py`)**: Predicts outcomes and assesses risk for individual records using the saved session model.
- **A/B Testing Analysis (`services/ab_testing.py`)**: Compares two cohorts via two-sample t-tests and relative uplift metrics.
- **Interactive Web UI (`streamlit_app.py`)**: Complete Streamlit dashboard with KPI cards, charts viewer, interactive manual prediction form, and PDF export.

### Planned Features
- Polite rejection of non-business datasets (e.g., computer vision or NLP text files).
- Interactive conversational AI data agent (chat with your dataset).
- Advanced automated segmentation and cohort retention heatmaps.

---

## 🏗️ Architecture

```text
[ User / Browser / Streamlit ]
              │
              ▼  (HTTP Multipart Upload: CSV / Excel / JSON)
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (app/main.py)               │
│                                                                 │
│   1. Upload Safeguards (Extension check + 20MB Size limit)      │
│   2. Inspection (Type inference, target detection)              │
│   3. Auto Clean (Deduplication, type fix, missing imputation)   │
│   4. Auto EDA (8 charts saved to /charts/{session_id})          │
│   5. Auto Model (Logistic/Ridge + Random Forest + XGBoost)      │
│   6. AI Insights (Groq LLM with offline rule-based fallback)     │
│   7. ReportLab PDF Export (/generate_report)                    │
└─────────────────────────────────────────────────────────────────┘
              │                           │
              ▼                           ▼
      [ Static Charts ]           [ Serialized Models ]
     charts/{session_id}/          models/{session_id}_model.pkl
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- `pip` and virtual environment support

### 2. Clone and Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Rohitkushwaha1310/AnchorIQ-.git
cd AnchorIQ

# Create and activate a clean virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
Install dependencies from the UTF-8 `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` (optional):
```ini
# Optional: Free tier Groq API key (https://console.groq.com)
GROQ_API_KEY=your_key_here

# Maximum upload limit in Megabytes (default: 20)
MAX_UPLOAD_SIZE_MB=20
```
> **Note**: If `GROQ_API_KEY` is left blank, AnchorIQ will automatically use its internal rule-based business insights engine.

---

## 🏃 Running the Application

### Start the FastAPI Backend
```bash
uvicorn app.main:app --reload --port 8000
```
- API Base URL: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`

### Start the Streamlit UI
In a separate terminal window (with `.venv` activated):
```bash
streamlit run streamlit_app.py
```
- Streamlit UI: `http://localhost:8501`

---

## 📡 API Reference Summary

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/health` | Healthcheck and API version | `200` |
| `POST` | `/analyze` | Upload file (`csv`, `xlsx`, `xls`, `json`) and execute full BI pipeline | `200`, `400` (bad format/empty), `413` (too large), `500` |
| `POST` | `/generate_report` | Run pipeline and stream downloadable PDF report | `200` (PDF binary), `400`, `413`, `500` |
| `POST` | `/predict` | Run manual single-row prediction against a session model | `200`, `404` (model not found), `500` |
| `POST` | `/ab_test` | Run statistical two-sample t-test between group A and B | `200`, `400`, `500` |
| `GET` | `/charts/{session_id}/{filename}` | Static file endpoint for generated charts | `200`, `404` |

---

## 🧪 Testing

AnchorIQ comes with a comprehensive pytest test suite validating all service modules, FastAPI endpoints, edge cases, and upload safeguards.

Run all tests:
```bash
pytest tests/test_api_and_services.py
```

### Edge Cases Covered:
1. Normal small CSV datasets.
2. Datasets with missing values and duplicate rows.
3. Empty files (properly rejected with HTTP 400).
4. Single-column datasets (gracefully processed without server crashes).
5. Targets with only a single unique class (returns a clear explanatory error).
6. Small sample datasets (<30 rows).
7. End-to-end full dataset regression testing on `telco_churn.csv`.
8. Upload safeguards: rejection of disallowed file types (`.exe` with HTTP 400) and payloads exceeding the file limit (HTTP 413).

---

## ⚠️ Limitations
- **File Size**: Uploads exceeding 20 MB (configurable via `MAX_UPLOAD_SIZE_MB`) are rejected by design to prevent out-of-memory errors on free-tier deployments.
- **Supervised ML Requirements**: Predictive modeling requires at least 15 valid sample rows and at least 2 distinct outcome classes in the target column.
- **LLM Rate Limits**: Free Groq API keys operate under standard free-tier rate limits. When limits are exceeded or keys are absent, AnchorIQ falls back to deterministic rule-based insights without interruption.
