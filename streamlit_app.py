
import streamlit as st
import requests
import pandas as pd


st.set_page_config(
    page_title="AnchorIQ",
    page_icon ="",
    layout    ="wide"
)


st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1B3A6B;
    text-align: center;
}
.sub-title {
    font-size: 1rem;
    color: #555555;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f0f4f8;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1B3A6B;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🧠 AnchorIQ</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Autonomous Data Analysis Platform — Upload any CSV, get instant ML insights!</p>',
            unsafe_allow_html=True)
st.divider()

#  SIDEBAR 
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=AnchorIQ",
             width=150)
    st.markdown("### ⚙️ Settings")
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000")
    st.divider()
    st.markdown("### 📌 How it works")
    st.markdown("""
    1. 📂 Upload any CSV file
    2. 🚀 Click Analyze
    3. 📊 Get instant insights:
       - Data inspection
       - Auto cleaning
       - EDA charts
       - ML model
       - AI insights
    """)
    st.divider()
    st.markdown("Built with ❤️ by **Rohit Kushwaha**")

#  UPLOAD SECTION 
col1, col2, col3 = st.columns([1,2,1])
with col2:
    uploaded = st.file_uploader(
    "📂 Upload your data file",
    type=['csv', 'xlsx', 'xls', 'json'],
    help="Supports CSV, Excel (.xlsx/.xls) and JSON!")

    if uploaded:
        st.success(f"✅ File ready: **{uploaded.name}**")
        df_preview = pd.read_csv(uploaded)
        uploaded.seek(0)

        with st.expander("👀 Preview Data"):
            st.dataframe(df_preview.head(5))
            st.caption(f"Shape: {df_preview.shape}")

        analyze_btn = st.button(
            "🚀 Analyze Now!",
            type ="primary",
            use_container_width=True)

#  ANALYSIS 
if uploaded and analyze_btn:
    with st.spinner("🔄 AnchorIQ is analyzing your data... Please wait!"):
        try:
            response = requests.post(
                f"{api_url}/analyze",
                files={"file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    "text/csv")},
                timeout=120
            )
            result = response.json()
        except Exception as e:
            st.error(f"❌ API Error: {e}")
            st.stop()

    st.success("✅ Analysis Complete!")
    st.balloons()

    #  KPI CARDS 
    insp = result.get('inspection', {})
    ml   = result.get('model_results', {})

    st.markdown("## 📊 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📋 Rows",     f"{insp.get('rows',0):,}")
    c2.metric("📊 Columns",  insp.get('columns',0))
    c3.metric("🎯 Target",   insp.get('target_column','N/A'))
    c4.metric("🔍 Nulls",    insp.get('missing_total',0))
    c5.metric("🔄 Dupes",    insp.get('duplicates',0))

    st.divider()

    # TABS 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA Charts",
        "🤖 ML Results",
        "💡 AI Insights",
        "🧹 Cleaning Report",
        "🔍 Data Stats"
    ])

    #  TAB 1: EDA CHARTS 
    with tab1:
        st.subheader("📊 Exploratory Data Analysis")
        charts = result.get('chart_urls', [])
        if charts:
            col1, col2 = st.columns(2)
            for i, url in enumerate(charts):
                with col1 if i % 2 == 0 else col2:
                    try:
                        st.image(url, use_column_width=True)
                    except:
                        st.warning(f"Chart {i+1} unavailable")
        else:
            st.info("No charts generated")

    #  TAB 2: ML RESULTS 
    with tab2:
        st.subheader("🤖 ML Model Results")
        if ml and 'error' not in ml:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Best Model",
                       ml.get('best_model','—'))
            m2.metric("📈 AUC Score",
                       ml.get('auc','—'))
            m3.metric("✅ Accuracy",
                       f"{ml.get('accuracy','—')}%")
            m4.metric("🔄 CV AUC",
                       ml.get('cv_mean','—'))

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Model Comparison")
                comparison = pd.DataFrame({
                    'Model': [
                        'Logistic Regression',
                        'Random Forest'],
                    'AUC': [
                        ml.get('lr_auc','—'),
                        ml.get('rf_auc','—')],
                    'Winner': [
                        '🏆' if ml.get('best_model')
                        =='Logistic Regression' else '',
                        '🏆' if ml.get('best_model')
                        =='Random Forest' else '']
                })
                st.dataframe(comparison,
                             hide_index=True)

            with col2:
                st.markdown("### Training Details")
                st.info(f"""
**Training samples:** {ml.get('train_samples',0):,}
**Testing samples:** {ml.get('test_samples',0):,}
**Features used:** {ml.get('n_features',0)}
**CV Std Dev:** {ml.get('cv_std','—')}
                """)
        else:
            st.warning("⚠️ ML model could not be trained. "
                      "No suitable target column found.")

    #  TAB 3: AI INSIGHTS 
    with tab3:
        st.subheader("💡 AI Business Insights")
        insights = result.get('insights','')
        if insights:
            st.markdown(insights)
        else:
            st.info("No insights generated")

    #  TAB 4: CLEANING REPORT 
    with tab4:
        st.subheader("🧹 Data Cleaning Report")
        clean = result.get('cleaning', {})

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Shape",
                    str(clean.get('original_shape','')))
        col2.metric("Cleaned Shape",
                    str(clean.get('cleaned_shape','')))
        col3.metric("Nulls Remaining",
                    clean.get('nulls_remaining',0))

        steps = clean.get('steps_applied', [])
        if steps:
            st.markdown("### ✅ Steps Applied")
            for step in steps:
                st.success(f"✅ {step}")
        else:
            st.info("Data was already clean!")

    # - TAB 5: DATA STATS 
    with tab5:
        st.subheader("🔍 Statistical Summary")
        stats = insp.get('stats', {})
        if stats:
            stats_df = pd.DataFrame(stats).T
            st.dataframe(stats_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Numerical Columns")
            for col in insp.get('numerical_cols', []):
                st.code(col)
        with col2:
            st.markdown("### 📝 Categorical Columns")
            for col in insp.get('categorical_cols', [])[:10]:
                st.code(col)