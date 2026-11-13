"""
AnchorIQ v2.0 — Autonomous Data Analysis & Predictive Machine Learning Platform
Streamlit Frontend — Professional 3D Spatial UI & Hybrid Autonomous Engine
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import io
import os
import json
import uuid

# Import internal services for seamless zero-downtime execution
from services.inspection import inspect_dataset
from services.cleaning   import auto_clean
from services.eda        import auto_eda
from services.modeling   import auto_model
from services.insights   import generate_insights
from services.pdf_report import generate_pdf_report


def _safe_streamlit_call(func, *args, **kwargs):
    """Drop use_container_width for older Streamlit versions that do not support it."""
    try:
        return func(*args, **kwargs)
    except TypeError:
        safe_kwargs = {k: v for k, v in kwargs.items() if k != "use_container_width"}
        return func(*args, **safe_kwargs)


def safe_image(path, **kwargs):
    return _safe_streamlit_call(st.image, path, **kwargs)


def safe_dataframe(*args, **kwargs):
    return _safe_streamlit_call(st.dataframe, *args, **kwargs)


def safe_bar_chart(*args, **kwargs):
    return _safe_streamlit_call(st.bar_chart, *args, **kwargs)


def safe_button(*args, **kwargs):
    return _safe_streamlit_call(st.button, *args, **kwargs)


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnchorIQ — Autonomous Data Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional 3D Spatial & Glassmorphism Styling ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Deep Spatial Canvas ── */
.stApp {
    background: radial-gradient(circle at 20% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 80% 85%, rgba(16, 185, 129, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.07) 0%, transparent 60%),
                linear-gradient(145deg, #070a12 0%, #0c1222 45%, #080d1a 100%);
    min-height: 100vh;
    color: #e2e8f0;
}

/* ── 3D Hero Billboard ── */
.hero-3d {
    position: relative;
    padding: 3rem 2rem 2.2rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 24px 48px -12px rgba(0, 0, 0, 0.6),
                0 4px 0 0 rgba(255, 255, 255, 0.08) inset,
                0 0 35px rgba(99, 102, 241, 0.2);
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(16px);
    transform: perspective(1000px) translateZ(0);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.hero-3d:hover {
    transform: perspective(1000px) translateY(-2px);
    box-shadow: 0 32px 64px -14px rgba(0, 0, 0, 0.75),
                0 4px 0 0 rgba(255, 255, 255, 0.12) inset,
                0 0 45px rgba(99, 102, 241, 0.3);
}

.hero-title {
    font-size: 3.4rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #ffffff 10%, #93c5fd 40%, #c084fc 75%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.15;
    text-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
}

.hero-sub {
    font-size: 1.12rem;
    color: rgba(226, 232, 240, 0.75);
    margin-top: 0.6rem;
    font-weight: 400;
    max-width: 780px;
    margin-left: auto;
    margin-right: auto;
}

.badge-3d {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.25));
    color: #a5b4fc;
    border: 1px solid rgba(165, 180, 252, 0.35);
    border-radius: 999px;
    padding: 0.3rem 1.1rem;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 1rem;
    letter-spacing: 0.06em;
    box-shadow: 0 8px 16px -4px rgba(99, 102, 241, 0.3), 0 2px 0 0 rgba(255, 255, 255, 0.15) inset;
}

/* ── 3D Floating KPI Tiles ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.kpi-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 18px;
    padding: 1.25rem 1rem;
    text-align: center;
    box-shadow: 0 16px 30px -10px rgba(0, 0, 0, 0.5),
                0 1px 0 0 rgba(255, 255, 255, 0.1) inset;
    transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1);
    transform: perspective(800px) translateZ(0);
}

.kpi-card:hover {
    transform: perspective(800px) translateY(-5px) translateZ(8px);
    border-color: rgba(99, 102, 241, 0.45);
    box-shadow: 0 24px 40px -12px rgba(99, 102, 241, 0.35),
                0 1px 0 0 rgba(255, 255, 255, 0.2) inset;
}

.kpi-val {
    font-size: 1.9rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.5px;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-lbl {
    font-size: 0.74rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-top: 0.35rem;
}

/* ── 3D Card Panels ── */
.panel-3d {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 20px 38px -12px rgba(0, 0, 0, 0.55),
                0 1px 0 0 rgba(255, 255, 255, 0.1) inset;
    backdrop-filter: blur(14px);
    margin-bottom: 1.5rem;
}

/* ── Executive AI Insight Box ── */
.insight-container {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(99, 102, 241, 0.09) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4),
                0 1px 0 0 rgba(255, 255, 255, 0.12) inset;
    line-height: 1.75;
    color: #f1f5f9;
}

.insight-container h2 {
    color: #6ee7b7;
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(110, 231, 183, 0.2);
    padding-bottom: 0.3rem;
}

.insight-container strong {
    color: #93c5fd;
}

/* ── 3D Tactile Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.75rem 1.6rem !important;
    box-shadow: 0 12px 24px -6px rgba(99, 102, 241, 0.5),
                0 3px 0 0 #3730a3 inset,
                0 -3px 0 0 rgba(0,0,0,0.3) inset !important;
    transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    letter-spacing: 0.02em !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 18px 30px -6px rgba(124, 58, 237, 0.65),
                0 3px 0 0 #4338ca inset !important;
}

.stButton > button:active {
    transform: translateY(2px) scale(0.99) !important;
    box-shadow: 0 6px 12px -2px rgba(99, 102, 241, 0.4),
                0 2px 0 0 rgba(0,0,0,0.4) inset !important;
}

/* ── 3D Inset File Uploader ── */
.stFileUploader {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.7), rgba(30, 41, 59, 0.4)) !important;
    border: 2px dashed rgba(99, 102, 241, 0.45) !important;
    border-radius: 20px !important;
    padding: 1rem !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) inset,
                0 0 25px rgba(99, 102, 241, 0.1) !important;
    transition: border-color 0.25s ease !important;
}

.stFileUploader:hover {
    border-color: #818cf8 !important;
}

/* ── 3D Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem !important;
    background: rgba(15, 23, 42, 0.6) !important;
    padding: 0.5rem !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3) inset !important;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #94a3b8 !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.2rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.3)) !important;
    color: #ffffff !important;
    box-shadow: 0 8px 20px -6px rgba(99, 102, 241, 0.4),
                0 1px 0 0 rgba(255, 255, 255, 0.2) inset !important;
    border: 1px solid rgba(165, 180, 252, 0.3) !important;
}

/* ── Sidebar Depth ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070c18 0%, #0c1427 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.4) !important;
}

/* ── Image 3D Frame ── */
.chart-frame {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 0.6rem;
    box-shadow: 0 16px 32px -8px rgba(0, 0, 0, 0.6),
                0 1px 0 0 rgba(255, 255, 255, 0.12) inset;
    margin-bottom: 1.2rem;
    transition: transform 0.25s ease;
}

.chart-frame:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 44px -10px rgba(99, 102, 241, 0.25);
}
</style>
""", unsafe_allow_html=True)


# ── Analysis Categories ───────────────────────────────────────────────────────
ANALYSIS_TYPES = {
    "auto"   : {"icon": "🤖", "label": "Auto Intelligence", "desc": "Autonomous problem type & driver detection"},
    "churn"  : {"icon": "🔄", "label": "Customer Churn",    "desc": "Retention, attrition risk & tenure drivers"},
    "sales"  : {"icon": "📈", "label": "Sales & Revenue",    "desc": "Revenue drivers, volumes & transaction value"},
    "fraud"  : {"icon": "🚨", "label": "Fraud & Anomaly",    "desc": "Transaction anomaly & risk score detection"},
    "hr"     : {"icon": "👥", "label": "Workforce & HR",     "desc": "Attrition, tenure, salary & department patterns"},
    "general": {"icon": "🔍", "label": "Exploratory EDA",    "desc": "Comprehensive statistical & predictive profiling"},
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='font-size: 2.8rem; filter: drop-shadow(0 0 15px rgba(99,102,241,0.5));'>🧠</div>
        <div style='font-size: 1.4rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;'>AnchorIQ</div>
        <div style='font-size: 0.75rem; color: #94a3b8; font-weight: 500;'>v2.0 Spatial Edition</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Backend Connection Strategy
    st.markdown("#### ⚡ Execution Architecture")
    engine_mode = st.radio(
        "Engine Mode:",
        ["Hybrid (Auto-Detect)", "Direct Engine (Embedded)", "FastAPI Backend (REST)"],
        index=0,
        help="AnchorIQ can run via high-speed in-memory engine or communicate with a FastAPI REST server."
    )

    api_url = st.text_input("REST Endpoint", value="http://localhost:8000", help="FastAPI server URL")

    st.divider()

    st.markdown("#### 🚀 Pipeline Features")
    st.markdown("""
    - **1. Smart Imputation**: Type-safe missing value resolution
    - **2. Spatial EDA**: 6 automated high-resolution visualizations
    - **3. ML AutoML**: Ridge/Logistic, Random Forest, XGBoost
    - **4. Groq Intelligence**: Executive CEO summary & roadmap
    - **5. Export PDF**: Instant executive downloadable report
    """)

    st.divider()
    st.caption("Crafted by **Rohit Kushwaha** | AnchorIQ v2.0")


# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-3d">
    <h1 class="hero-title">AnchorIQ Autonomous Studio</h1>
    <p class="hero-sub">Drop any dataset to generate end-to-end exploratory analysis, predictive machine learning benchmarks, and executive AI intelligence reports.</p>
    <div class="badge-3d">✨ 3D Spatial Intelligence &bull; Production Ready</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD & DATA CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
col_upload, col_controls = st.columns([1.2, 1], gap="large")

with col_upload:
    st.markdown("### 📂 1. Import Dataset")
    uploaded = st.file_uploader(
        "Drop dataset (CSV, Excel, or JSON)",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    df_preview = None
    target_column_choice = None

    if uploaded:
        try:
            fname = uploaded.name.lower()
            if fname.endswith(".csv"):
                df_preview = pd.read_csv(uploaded, encoding_errors="replace")
            elif fname.endswith((".xlsx", ".xls")):
                df_preview = pd.read_excel(uploaded)
            elif fname.endswith(".json"):
                df_preview = pd.read_json(uploaded)
            uploaded.seek(0)

            st.success(f"✅ **{uploaded.name}** loaded — {df_preview.shape[0]:,} rows &times; {df_preview.shape[1]} features")

            with st.expander("🔍 Dataset Quick Inspection", expanded=True):
                safe_dataframe(df_preview.head(5), use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")
            uploaded.seek(0)

with col_controls:
    st.markdown("### 🎯 2. Select Objective")
    analysis_type = st.radio(
        "Analysis Type",
        options=list(ANALYSIS_TYPES.keys()),
        format_func=lambda k: f"{ANALYSIS_TYPES[k]['icon']}  {ANALYSIS_TYPES[k]['label']}",
        label_visibility="collapsed",
    )

    t_info = ANALYSIS_TYPES[analysis_type]
    st.markdown(f"""
    <div style='background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25);
                border-radius: 12px; padding: 0.8rem 1.1rem; margin-top: 0.5rem; margin-bottom: 1rem;'>
        <div style='font-size: 1.1rem; font-weight: 700; color: #c7d2fe;'>{t_info['icon']} {t_info['label']}</div>
        <div style='font-size: 0.82rem; color: #94a3b8; margin-top: 0.2rem;'>{t_info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic target selector if data is loaded
    if df_preview is not None and not df_preview.empty:
        # Check auto-guessed target
        temp_insp = inspect_dataset(df_preview, analysis_type=analysis_type)
        auto_t = temp_insp.get("target_column")
        all_cols = list(df_preview.columns)

        default_idx = all_cols.index(auto_t) if auto_t in all_cols else 0
        target_column_choice = st.selectbox(
            "Target Column (Auto-Selected, customizable):",
            options=all_cols,
            index=default_idx,
            help="Choose the outcome variable you want to predict or analyze."
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION BUTTON & ENGINE DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    btn_disabled = uploaded is None
    analyze_btn = safe_button(
        "🚀 Launch Autonomous Analysis",
        type="primary",
        use_container_width=True,
        disabled=btn_disabled,
    )
    if btn_disabled:
        st.caption("<div style='text-align:center;color:#64748b;'>Upload a CSV, Excel, or JSON file to unlock analysis</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION (HYBRID ZERO-FAILURE)
# ══════════════════════════════════════════════════════════════════════════════
if uploaded and analyze_btn:
    progress_bar = st.progress(0, text="🔄 Initializing AnchorIQ engine...")
    session_id = str(uuid.uuid4())[:8]

    # Attempt execution
    use_api = False
    if "FastAPI" in engine_mode:
        use_api = True
    elif "Hybrid" in engine_mode:
        try:
            h_resp = requests.get(f"{api_url}/health", timeout=1.5)
            if h_resp.status_code == 200:
                use_api = True
        except Exception:
            use_api = False

    result = None

    if use_api:
        progress_bar.progress(25, text="🌐 Dispatching to FastAPI REST Backend...")
        try:
            uploaded.seek(0)
            data_payload = {"analysis_type": analysis_type}
            if target_column_choice:
                data_payload["target_column"] = target_column_choice

            api_res = requests.post(
                f"{api_url}/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
                data=data_payload,
                timeout=240,
            )
            result = api_res.json()
            result["engine_used"] = "FastAPI Backend (REST API)"
            progress_bar.progress(100, text="✅ Analysis successfully finalized via FastAPI!")

        except Exception as e:
            st.warning(f"⚠️ FastAPI connection issue ({e}). Seamlessly switching to Embedded Direct Engine...")
            use_api = False

    if not use_api:
        # Embedded Direct Engine Execution (Guaranteed Zero Network Failure)
        progress_bar.progress(20, text="🧹 Cleaning and imputing dataset...")
        uploaded.seek(0)
        fname = uploaded.name.lower()
        if fname.endswith(".csv"):
            df_raw = pd.read_csv(uploaded, encoding_errors="replace")
        elif fname.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded)
        elif fname.endswith(".json"):
            df_raw = pd.read_json(uploaded)
        else:
            df_raw = pd.DataFrame()

        # Step 1: Inspect
        inspection = inspect_dataset(df_raw, analysis_type=analysis_type, target_column=target_column_choice)
        target = inspection["target_column"]

        # Step 2: Clean
        df_clean, clean_report = auto_clean(df_raw.copy())

        # Step 3: EDA
        progress_bar.progress(50, text="📊 Generating 3D-styled Exploratory Visualizations...")
        chart_dir = f"charts/{session_id}"
        chart_paths = auto_eda(df_clean, target=target, analysis_type=analysis_type, save_dir=chart_dir)

        # Step 4: ML Modeling
        progress_bar.progress(75, text="🤖 Training Ridge, Random Forest & XGBoost...")
        model_results = {}
        if target:
            model_results = auto_model(df_clean.copy(), target=target, analysis_type=analysis_type)

        # Step 5: Insights
        progress_bar.progress(90, text="💡 Formulating Executive AI Insights...")
        insights = generate_insights(
            inspection=inspection,
            model_results=model_results,
            target=target or "outcome",
            analysis_type=analysis_type,
            feature_importance=model_results.get("feature_importance", {})
        )

        result = {
            "filename"      : uploaded.name,
            "session_id"    : session_id,
            "analysis_type" : analysis_type,
            "inspection"    : inspection,
            "cleaning"      : clean_report,
            "chart_paths"   : chart_paths,
            "chart_urls"    : chart_paths,
            "model_results" : model_results,
            "insights"      : insights,
            "engine_used"   : "⚡ Direct Autonomous Engine (High Performance)",
        }
        progress_bar.progress(100, text="✅ Complete!")

    if result and "error" not in result:
        st.session_state["result"] = result
        st.session_state["analysis_type"] = analysis_type
        st.session_state["uploaded_file_bytes"] = uploaded.getvalue()
        st.session_state["uploaded_file_name"] = uploaded.name
        st.success("🎉 Analysis successfully generated!")
        st.balloons()
    elif result and "error" in result:
        st.error(f"❌ Analysis failed: {result['error']}")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "result" in st.session_state:
    res           = st.session_state["result"]
    analysis_type = st.session_state.get("analysis_type", "auto")
    insp          = res.get("inspection", {})
    ml            = res.get("model_results", {})
    clean         = res.get("cleaning", {})
    chart_paths   = res.get("chart_paths", [])
    engine_used   = res.get("engine_used", "Autonomous Engine")

    t_icon = ANALYSIS_TYPES.get(analysis_type, {}).get("icon", "🤖")
    t_name = ANALYSIS_TYPES.get(analysis_type, {}).get("label", "Analysis")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;'>
        <div style='font-size:1.6rem; font-weight:800; color:#f8fafc;'>
            {t_icon} {t_name} Results
        </div>
        <div style='background:rgba(99,102,241,0.2); border:1px solid rgba(165,180,252,0.3);
                    border-radius:999px; padding:0.25rem 0.9rem; font-size:0.8rem; color:#a5b4fc; font-weight:600;'>
            {engine_used}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Tiles Grid ────────────────────────────────────────────────────────
    is_reg = ml.get("problem_type") == "regression"
    if is_reg:
        score_num = ml.get("r2_score") if ml.get("r2_score") is not None else ml.get("r2")
        kpi_score_val = f"{score_num:.4f}" if isinstance(score_num, (int, float)) else str(score_num or "—")
        kpi_score_lbl = "R² Champion Score"
        mae_num = ml.get("mae") if ml.get("mae") is not None else ml.get("rmse")
        kpi_sec_val   = f"{mae_num:.4f}" if isinstance(mae_num, (int, float)) else str(mae_num or "—")
        kpi_sec_lbl   = "MAE Error"
    else:
        auc_num = ml.get("auc") if ml.get("auc") is not None else ml.get("auc_score")
        kpi_score_val = f"{auc_num:.4f}" if isinstance(auc_num, (int, float)) else str(auc_num or "—")
        kpi_score_lbl = "AUC-ROC Score"
        acc_num = ml.get("accuracy") if ml.get("accuracy") is not None else ml.get("accuracy_score")
        kpi_sec_val   = f"{acc_num}%" if acc_num is not None else "—"
        kpi_sec_lbl   = "Holdout Accuracy"

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-val">{insp.get('rows', 0):,}</div>
            <div class="kpi-lbl">Total Records</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{insp.get('columns', 0)}</div>
            <div class="kpi-lbl">Features</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="font-size:1.4rem;">{str(insp.get('target_column', 'N/A'))[:13]}</div>
            <div class="kpi-lbl">Target Outcome</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="font-size:1.35rem; color:#a78bfa;">{ml.get('best_model', 'N/A')}</div>
            <div class="kpi-lbl">Best ML Model</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color:#34d399;">{kpi_score_val}</div>
            <div class="kpi-lbl">{kpi_score_lbl}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val" style="color:#38bdf8;">{kpi_sec_val}</div>
            <div class="kpi-lbl">{kpi_sec_lbl}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── 8 Interactive Workspace Tabs ──────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 3D EDA Charts",
        "🤖 AutoML Studio",
        "💡 AI Intelligence",
        "🧹 Data Quality",
        "🔍 Feature Statistics",
        "📄 Executive PDF Report",
        "🧪 A/B Test",
        "🎯 Predict",
        "💬 Ask AnchorIQ",
    ])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: 3D EDA CHARTS
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 📊 Automated Exploratory Data Visualizations")
        chart_labels = [
            "Feature Distributions & Histograms",
            "Correlation Matrix & Co-linearity",
            "Target Distribution & Imbalance",
            "Categorical Impact & Conversion Rates",
            "Numerical Box Plots & Quartiles",
            "Domain Risk & Specialized Visualizations",
        ]

        valid_charts = [p for p in chart_paths if os.path.exists(p)]
        if valid_charts:
            col_c1, col_c2 = st.columns(2, gap="large")
            for i, p in enumerate(valid_charts):
                container = col_c1 if i % 2 == 0 else col_c2
                lbl = chart_labels[i] if i < len(chart_labels) else f"Visual Analysis {i+1}"
                with container:
                    st.markdown(f"**{i+1}. {lbl}**")
                    st.markdown(f"<div class='chart-frame'>", unsafe_allow_html=True)
                    safe_image(p, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No charts generated.")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: AUTOML STUDIO
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🤖 Predictive Machine Learning Engine")

        if ml and "error" not in ml:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("🏆 Champion Model", ml.get("best_model", "—"))
            if is_reg:
                r2_val = ml.get("r2_score") if ml.get("r2_score") is not None else ml.get("r2", "—")
                m_col2.metric("📈 R² Fit Score", f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val))
                mae_val = ml.get("mae") if ml.get("mae") is not None else ml.get("rmse", "—")
                m_col3.metric("📉 MAE Error", f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else str(mae_val))
                m_col4.metric("🔄 5-Fold CV R²", f"{ml.get('cv_mean','—')} ± {ml.get('cv_std','0')}")
            else:
                auc_val = ml.get("auc") if ml.get("auc") is not None else ml.get("auc_score", "—")
                m_col2.metric("📈 AUC-ROC", f"{auc_val:.4f}" if isinstance(auc_val, (int, float)) else str(auc_val))
                m_col3.metric("🎯 Accuracy", f"{ml.get('accuracy', '—')}%")
                m_col4.metric("🔄 5-Fold CV Score", f"{ml.get('cv_mean','—')} ± {ml.get('cv_std','0')}")

            st.divider()

            col_cmp, col_details = st.columns([1.2, 1], gap="large")

            with col_cmp:
                st.markdown("#### 🏆 Model Tournament Benchmark")
                best_model_name = ml.get("best_model", "")
                metric_name = ml.get("metric", "AUC-ROC" if not is_reg else "R²")
                card_scores = ml.get("model_card", {}).get("cv_scores", {})

                if card_scores:
                    cmp_rows = []
                    for model_name, score_info in card_scores.items():
                        if isinstance(score_info, dict):
                            if "mean" in score_info:
                                mean_val = score_info["mean"]
                                std_val = score_info.get("std", 0.0)
                                score_str = f"{mean_val:.4f} ± {std_val:.4f}"
                            elif "skipped" in score_info:
                                score_str = f"Skipped: {score_info['skipped']}"
                            elif "error" in score_info:
                                score_str = f"Error: {score_info['error']}"
                            else:
                                score_str = "—"
                        else:
                            score_str = str(score_info)

                        is_winner = (model_name == best_model_name)
                        cmp_rows.append({
                            "Model": model_name,
                            f"CV Score ({metric_name})": score_str,
                            "Status": "🏆 Champion" if is_winner else ""
                        })
                    safe_dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
                else:
                    # Fallback to direct keys
                    cmp_data = [
                        {"Model": "Logistic Regression" if not is_reg else "Ridge", "Score": ml.get("lr_score", "—"), "Status": "🏆 Champion" if best_model_name in ["Logistic Regression", "Ridge"] else ""},
                        {"Model": "Decision Tree", "Score": ml.get("dt_score", "—"), "Status": "🏆 Champion" if best_model_name == "Decision Tree" else ""},
                        {"Model": "Random Forest", "Score": ml.get("rf_score", "—"), "Status": "🏆 Champion" if best_model_name == "Random Forest" else ""},
                        {"Model": "Extra Trees", "Score": ml.get("et_score", "—"), "Status": "🏆 Champion" if best_model_name == "Extra Trees" else ""},
                        {"Model": "XGBoost", "Score": ml.get("xgb_score", "—"), "Status": "🏆 Champion" if best_model_name == "XGBoost" else ""},
                        {"Model": "Ensemble", "Score": ml.get("ensemble_score", "—"), "Status": "🏆 Champion" if best_model_name == "Ensemble" else ""},
                    ]
                    safe_dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

            with col_details:
                st.markdown("#### 📋 Architecture Details")
                st.markdown(f"""
                - **Champion Model:** `{best_model_name}`
                - **Training Instances:** `{ml.get('train_samples', 0):,}`
                - **Evaluation Holdout:** `{ml.get('test_samples', 0):,}`
                - **Active Engineered Features:** `{ml.get('n_features', 0)}`
                - **CV Stability (Std Dev):** `{ml.get('cv_std', '0')}`
                - **Problem Formulation:** `{ml.get('problem_type', 'N/A').title()}`
                - **Reliability Verdict:** `{str(ml.get('model_card', {}).get('reliability_verdict', 'N/A')).title()}`
                """)
                why_won = ml.get("model_card", {}).get("why_it_won")
                if why_won:
                    st.info(f"**Selection Rationale:** {why_won}")
                ens_details = ml.get("model_card", {}).get("ensemble_details")
                if ens_details:
                    st.caption(f"**Ensemble Audit:** {ens_details}")

            # Shallow Decision Tree rules
            rules = ml.get("decision_rules") or ml.get("model_card", {}).get("decision_rules")
            if rules:
                with st.expander("🌲 Shallow Decision Tree Rules (Explainability)", expanded=False):
                    st.code(rules, language="text")

            # Feature Importance
            fi = ml.get("feature_importance", {})
            if fi:
                st.divider()
                st.markdown("#### 🔑 Influential Feature Drivers")
                fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Relative Importance": list(fi.values())})
                fi_df = fi_df.sort_values("Relative Importance", ascending=True).tail(12)
                safe_bar_chart(fi_df.set_index("Feature"), use_container_width=True)

        elif ml and "error" in ml:
            st.warning(f"⚠️ Machine Learning could not complete: {ml['error']}")
        else:
            st.info("No suitable outcome variable was selected for predictive modeling.")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: AI INTELLIGENCE
    # ──────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 💡 Strategic AI Executive Intelligence")
        insights = res.get("insights", "")
        if insights:
            st.markdown(f"<div class='insight-container'>{insights}</div>", unsafe_allow_html=True)
        else:
            st.info("Insights generation in progress...")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 4: DATA QUALITY
    # ──────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🧹 Data Cleaning & Integrity Audit")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("📦 Raw Dimensions", str(clean.get("original_shape", "N/A")))
        q2.metric("✨ Processed Dimensions", str(clean.get("cleaned_shape", "N/A")))
        q3.metric("🔍 Missing Cells Fixed", str(insp.get("missing_total", 0)))
        q4.metric("🧹 Duplicates Purged", str(insp.get("duplicates", 0)))

        # Quality index score
        rows = insp.get("rows", 1)
        nulls = insp.get("missing_total", 0)
        dupes = insp.get("duplicates", 0)
        quality_score = max(0, 100 - (nulls + dupes) / max(rows, 1) * 100)
        q_label = "🟢 High Integrity" if quality_score >= 95 else "🟡 Good Integrity" if quality_score >= 80 else "🔴 Needs Review"

        st.markdown(f"**Data Integrity Index:** `{quality_score:.1f}%` — {q_label}")
        st.progress(quality_score / 100)

        st.divider()
        steps = clean.get("steps_applied", [])
        if steps:
            st.markdown("#### ✅ Automated Preprocessing Execution Log")
            for s in steps:
                st.success(f"✔️ {s}")
        else:
            st.info("✨ Data was completely clean — no transformations required!")

        missing_cols = insp.get("missing_cols", {})
        if missing_cols:
            st.divider()
            st.markdown("#### 🔍 Missing Values Resolved by Column")
            safe_dataframe(pd.DataFrame(list(missing_cols.items()), columns=["Column", "Imputed Count"]), use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 5: FEATURE STATISTICS
    # ──────────────────────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 🔍 Statistical Feature Summary")
        stats = insp.get("stats", {})
        if stats:
            safe_dataframe(pd.DataFrame(stats).T, use_container_width=True)

        st.divider()
        c_num, c_cat = st.columns(2, gap="large")
        with c_num:
            st.markdown("#### 🔢 Numerical Variables")
            nums = insp.get("numerical_cols", [])
            st.write(nums if nums else "None detected")

        with c_cat:
            st.markdown("#### 📝 Categorical Variables")
            cats = insp.get("categorical_cols", [])
            st.write(cats if cats else "None detected")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 6: EXECUTIVE PDF REPORT
    # ──────────────────────────────────────────────────────────────────────────
    with tab6:
        st.markdown("### 📄 Export Executive PDF Intelligence Report")
        st.markdown("""
        Generate a comprehensive, boardroom-ready PDF document including:
        - **Executive Cover & Metadata**
        - **AI Strategic Roadmaps & Business Recommendations**
        - **Dataset Quality Metrics & Cleaning Log**
        - **AutoML Tournament Scores & Influential Drivers**
        - **All 6 Visual EDA Visualizations Embedded**
        - **Detailed Statistical Summaries**
        """)

        # Generate on demand
        pdf_path = f"reports/AnchorIQ_{analysis_type}_{res.get('session_id', 'export')}.pdf"

        if safe_button("📄 Render & Download Executive PDF", type="primary", use_container_width=True):
            with st.spinner("🔄 Building high-resolution PDF document..."):
                try:
                    generated_pdf = generate_pdf_report(
                        result=res,
                        chart_paths=chart_paths,
                        output_path=pdf_path,
                    )
                    if os.path.exists(generated_pdf):
                        with open(generated_pdf, "rb") as f:
                            pdf_bytes = f.read()

                        st.success("✅ Executive PDF generated successfully!")
                        st.download_button(
                            label="⬇️ Click Here to Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"AnchorIQ_{analysis_type}_Executive_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

        st.divider()
        st.markdown("#### 📋 Report Executive Summary Preview")
        summary_items = {
            "📁 Dataset Name"   : res.get("filename", "N/A"),
            "🔬 Objective"      : t_name,
            "📊 Record Volume"  : f"{insp.get('rows', 0):,}",
            "🎯 Target Variable": insp.get("target_column", "N/A"),
            "🏆 Winning Model"  : ml.get("best_model", "N/A"),
            "📈 Primary Score"  : kpi_score_val,
            "📊 Charts Included": f"{len(valid_charts)} figures",
        }
        for k, v in summary_items.items():
            k_col, v_col = st.columns([1, 2])
            k_col.markdown(f"**{k}**")
            v_col.markdown(str(v))

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 7: A/B TEST
    # ──────────────────────────────────────────────────────────────────────────
    with tab7:
        st.markdown("### 🧪 Statistical A/B Testing & Hypothesis Engine")
        st.markdown("Evaluate statistically significant differences across customer segments or experiment groups.")

        run_id_val = res.get("run_id") or res.get("session_id")
        current_df = st.session_state.get("df_clean") or st.session_state.get("df_raw")
        if current_df is None and "df_clean" in locals():
            current_df = df_clean

        col_names = list(current_df.columns) if current_df is not None else list(insp.get("stats", {}).keys())
        cat_candidates = insp.get("categorical_cols", []) or [c for c in col_names if c != insp.get("target_column")]
        num_candidates = insp.get("numerical_cols", []) or [c for c in col_names if c not in cat_candidates]

        col_ab1, col_ab2 = st.columns(2)
        with col_ab1:
            group_col_selected = st.selectbox(
                "Experiment / Group Column",
                cat_candidates if cat_candidates else col_names,
                key="tab7_group_col"
            )
        with col_ab2:
            metric_col_selected = st.selectbox(
                "Outcome / Metric Column",
                num_candidates if num_candidates else col_names,
                key="tab7_metric_col"
            )

        if safe_button("🔬 Run Statistical A/B Test", type="primary", use_container_width=True, key="tab7_run_btn"):
            with st.spinner("Executing statistical hypothesis tests..."):
                ab_result = None
                try:
                    payload = {
                        "run_id": run_id_val,
                        "group_column": group_col_selected,
                        "metric_column": metric_col_selected,
                    }
                    r_post = requests.post(f"{api_url}/ab_test", json=payload, timeout=8)
                    if r_post.status_code == 200:
                        ab_result = r_post.json()
                except Exception:
                    pass

                if not ab_result and current_df is not None:
                    from services.ab_testing import run_ab_test
                    try:
                        ab_result = run_ab_test(df=current_df, group_column=group_col_selected, metric_column=metric_col_selected)
                    except Exception as abe:
                        st.error(f"A/B test calculation error: {abe}")

                if ab_result:
                    is_sig = ab_result.get("significant", False)
                    p_val = ab_result.get("p_value", 1.0)
                    sig_badge = "🟢 Statistically Significant" if is_sig else "🔴 Not Significant"
                    sig_color = "#10b981" if is_sig else "#ef4444"

                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.04); border-left:4px solid {sig_color}; border-radius:12px; padding:1.2rem; margin:1rem 0;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <h4 style='margin:0; color:#ffffff;'>Test Method: {ab_result.get("test_type", "A/B Test")}</h4>
                            <span style='background:{sig_color}22; color:{sig_color}; border:1px solid {sig_color}55; padding:0.25rem 0.8rem; border-radius:999px; font-weight:700; font-size:0.85rem;'>
                                {sig_badge}
                            </span>
                        </div>
                        <p style='margin:0.8rem 0 0 0; color:#e2e8f0; font-size:1.05rem;'>{ab_result.get("plain_english_verdict")}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("P-Value", f"{p_val:.4f}", delta="p < 0.05" if is_sig else "p ≥ 0.05", delta_color="normal" if is_sig else "inverse")
                    m2.metric(ab_result.get("effect_size_label", "Effect Size"), f"{ab_result.get('effect_size', '—')}")
                    ci = ab_result.get("confidence_interval")
                    m3.metric("95% Confidence Interval", f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "N/A")
                    m4.metric("Metric Type", ab_result.get("metric_type", "N/A").title())

                    st.markdown("#### 👥 Group Sample Sizes & Breakdown")
                    st.json(ab_result.get("group_stats") or ab_result.get("group_sizes"))

                    if ab_result.get("warnings"):
                        for w in ab_result["warnings"]:
                            st.warning(f"⚠️ {w}")

                    if ab_result.get("notes"):
                        for n in ab_result["notes"]:
                            st.caption(f"📌 {n}")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 8: PREDICT
    # ──────────────────────────────────────────────────────────────────────────
    with tab8:
        st.markdown("### 🎯 Dynamic Interactive Model Predictor")
        st.markdown("Enter values for features below. The form is generated dynamically from the model's schema so it supports any dataset.")

        run_id_val = res.get("run_id") or res.get("session_id")
        schema = None
        try:
            r_schema = requests.get(f"{api_url}/predict/schema/{run_id_val}", timeout=5)
            if r_schema.status_code == 200:
                schema = r_schema.json()
        except Exception:
            pass

        if not schema:
            schema = ml.get("schema") or {}

        cols_schema = schema.get("columns") or schema.get("fields") or {}
        if not cols_schema:
            st.info("No interactive prediction schema available for this run.")
        else:
            with st.form(key=f"predict_form_{run_id_val}"):
                input_fields = {}
                col_names = list(cols_schema.keys())
                form_cols = st.columns(2)
                for idx, col_name in enumerate(col_names):
                    meta = cols_schema[col_name]
                    target_container = form_cols[idx % 2]
                    kind = meta.get("kind") if isinstance(meta, dict) else None

                    with target_container:
                        if kind == "categorical" or (isinstance(meta, dict) and "allowed_categories" in meta):
                            categories = meta.get("allowed_categories", [])
                            input_fields[col_name] = st.selectbox(
                                f"{col_name}",
                                options=categories,
                                key=f"form_input_{col_name}"
                            )
                        else:
                            min_val = float(meta.get("min", 0.0)) if isinstance(meta, dict) and meta.get("min") is not None else 0.0
                            max_val = float(meta.get("max", 100000.0)) if isinstance(meta, dict) and meta.get("max") is not None else 100000.0
                            default_val = (min_val + max_val) / 2.0 if min_val < max_val else min_val
                            input_fields[col_name] = st.number_input(
                                f"{col_name} (Range: [{min_val:g}, {max_val:g}])",
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                key=f"form_input_{col_name}"
                            )

                submitted_pred = st.form_submit_button("⚡ Run Prediction", type="primary", use_container_width=True)

            if submitted_pred:
                with st.spinner("Evaluating model pipeline & calculating decision drivers..."):
                    pred_res = None
                    try:
                        r_pred = requests.post(f"{api_url}/predict/{run_id_val}", json=input_fields, timeout=10)
                        if r_pred.status_code == 200:
                            pred_res = r_pred.json()
                        else:
                            st.error(f"Prediction failed ({r_pred.status_code}): {r_pred.text}")
                    except Exception:
                        pass

                    if not pred_res:
                        from services.predictor import predict_single
                        try:
                            pred_res = predict_single(session_id=run_id_val, input_data=input_fields)
                        except Exception as pe:
                            st.error(f"Prediction error: {pe}")

                    if pred_res:
                        st.divider()
                        st.markdown("### 🏆 Prediction Results")

                        risk_tier = pred_res.get("risk_label", "Medium")
                        risk_col = "#10b981" if risk_tier == "Low" else ("#f59e0b" if risk_tier == "Medium" else "#ef4444")

                        p_col1, p_col2, p_col3 = st.columns(3)
                        p_col1.metric("Predicted Output", str(pred_res.get("prediction", "N/A")))
                        prob_val = pred_res.get("calibrated_probability", pred_res.get("probability", 0))
                        p_col2.metric("Calibrated Probability", f"{float(prob_val):.1%}" if isinstance(prob_val, (int, float)) else str(prob_val))
                        p_col3.metric("Risk Label", risk_tier)

                        st.markdown(f"""
                        <div style='background:rgba(255,255,255,0.04); border-left:4px solid {risk_col}; border-radius:10px; padding:1rem; margin:1rem 0;'>
                            <strong style='color:#ffffff;'>Recommendation:</strong>
                            <p style='margin:0.4rem 0 0 0; color:#e2e8f0;'>{pred_res.get("recommendation", "")}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Reasons under the prediction
                        st.markdown("#### 🔍 Top 3 Explanatory Reasons")
                        reasons = pred_res.get("reasons", [])
                        if reasons:
                            for r_i, reason_text in enumerate(reasons[:3], 1):
                                st.markdown(f"**{r_i}.** {reason_text}")
                        else:
                            st.info("No specific explanatory drivers returned.")

                        if pred_res.get("shap_status"):
                            st.caption(f"ℹ️ Explanation Engine: {pred_res.get('shap_status')}")

    with tab9:
        st.markdown("### 💬 Ask AnchorIQ")
        st.caption("Answers use stored evidence from this analysis. Probability questions require customer feature inputs.")
        run_id_val = res.get("run_id")
        if not run_id_val:
            st.info("Run this analysis through the FastAPI backend to enable saved chat.")
        else:
            chat_key = f"anchoriq_chat_{run_id_val}"
            st.session_state.setdefault(chat_key, [])
            st.markdown("**Suggested questions**")
            suggestions = [
                "Why are customers leaving?",
                "What would a 10 percent discount do?",
                "What is the churn probability for this customer?",
            ]
            suggestion_cols = st.columns(3)
            selected_question = None
            for index, suggestion in enumerate(suggestions):
                if suggestion_cols[index].button(suggestion, key=f"chat_suggestion_{run_id_val}_{index}", use_container_width=True):
                    selected_question = suggestion

            customer_inputs_text = st.text_area(
                "Optional customer features (JSON, for probability questions)",
                placeholder='{"tenure": 12, "MonthlyCharges": 75, "Contract": "Month-to-month"}',
                key=f"chat_inputs_{run_id_val}",
            )
            typed_question = st.chat_input("Ask about this analysis")
            question = selected_question or typed_question
            for message in st.session_state[chat_key]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if question:
                inputs = None
                if customer_inputs_text.strip():
                    try:
                        inputs = json.loads(customer_inputs_text)
                        if not isinstance(inputs, dict):
                            raise ValueError("Expected a JSON object")
                    except (json.JSONDecodeError, ValueError) as exc:
                        st.error(f"Customer features must be a JSON object: {exc}")
                        inputs = None
                        question = None
                if question:
                    st.session_state[chat_key].append({"role": "user", "content": question})
                    try:
                        response = requests.post(
                            f"{api_url}/chat/{run_id_val}",
                            json={"question": question, "inputs": inputs},
                            timeout=30,
                        )
                        response.raise_for_status()
                        answer = response.json().get("answer", "I don't have data for that.")
                    except Exception as exc:
                        answer = f"Chat is unavailable: {exc}"
                    st.session_state[chat_key].append({"role": "assistant", "content": answer})
                    st.rerun()

# ── Empty State ───────────────────────────────────────────────────────────────
elif not uploaded:
    st.markdown("""
    <div style='text-align: center; padding: 4rem 1rem; color: rgba(255,255,255,0.4);'>
        <div style='font-size: 4rem; margin-bottom: 0.8rem; filter: drop-shadow(0 0 20px rgba(99,102,241,0.3));'>📂</div>
        <div style='font-size: 1.3rem; font-weight: 700; color: #cbd5e1;'>Upload any dataset above to begin</div>
        <div style='font-size: 0.9rem; color: #64748b; margin-top: 0.3rem;'>Supports CSV, Excel (.xlsx/.xls), and JSON files</div>
    </div>
    """, unsafe_allow_html=True)