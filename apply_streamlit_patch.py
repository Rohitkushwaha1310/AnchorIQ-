"""
Run ONCE from your project root:   python apply_streamlit_patch.py
- Makes a backup: streamlit_app.py.bak
- Wires the UI to the new engine (all file types, predictions, recommendations, interactive charts)
"""
import ast
import shutil
import sys

PATH = "streamlit_app.py"
src = open(PATH, encoding="utf-8").read().replace("\r\n", "\n")
if "run_analysis" in src:
    sys.exit("Already patched - nothing to do.")


def swap(old, new, count=1):
    global src
    if src.count(old) != count:
        sys.exit(f"Could not find the expected code block (found {src.count(old)}x):\n{old[:120]}...")
    src = src.replace(old, new)


# 1) imports
swap("from services.pdf_report import generate_pdf_report\n",
     "from services.pdf_report import generate_pdf_report\n"
     "from services.loader     import load_any, SUPPORTED_EXTENSIONS\n"
     "from services.profiler   import profile\n"
     "from services.pipeline   import run_analysis\n")

# 2) uploader accepts every supported type
swap('type=["csv", "xlsx", "xls", "json"],', 'type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],')

# 3) preview uses the universal loader
swap('''            fname = uploaded.name.lower()
            if fname.endswith(".csv"):
                df_preview = pd.read_csv(uploaded, encoding_errors="replace")
            elif fname.endswith((".xlsx", ".xls")):
                df_preview = pd.read_excel(uploaded)
            elif fname.endswith(".json"):
                df_preview = pd.read_json(uploaded)
            uploaded.seek(0)
''', '''            df_preview, _meta = load_any(uploaded.name, uploaded.getvalue())
            uploaded.seek(0)
''')

# 4) smarter target suggestion (with confidence)
swap('''        temp_insp = inspect_dataset(df_preview, analysis_type=analysis_type)
        auto_t = temp_insp.get("target_column")
''', '''        _, _pinfo = profile(df_preview, analysis_type)
        auto_t = _pinfo["target"]
        if auto_t is None:
            st.info("No clear outcome column found. AnchorIQ will look for segments, anomalies and trends - or pick a target below.")
        elif _pinfo["target_confidence"] < 0.5:
            st.warning(f"Suggested target '{auto_t}' is a low-confidence guess - please confirm it below.")
''')

# 5) direct engine -> one call to the shared pipeline
a = src.index("        # Embedded Direct Engine Execution")
b_marker = '        progress_bar.progress(100, text="✅ Complete!")\n'
b = src.index(b_marker, a) + len(b_marker)
src = src[:a] + '''        # Embedded Direct Engine (same code path as the FastAPI backend)
        progress_bar.progress(30, text="🔍 Reading, cleaning and analysing your data...")
        df_raw, _meta = load_any(uploaded.name, uploaded.getvalue())
        result = run_analysis(df_raw, uploaded.name, analysis_type, target_column_choice,
                              session_id, _meta["notes"])
        result["engine_used"] = "⚡ Direct Autonomous Engine"
        progress_bar.progress(100, text="✅ Complete!")
''' + src[b:]

# 6) two new tabs in front of the existing six
swap("    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([\n",
     "    tab_act, tab_viz, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([\n"
     '        "🎯 Actions & Predictions",\n        "📈 Interactive Charts",\n')

# 7) tab bodies (inserted right before the empty-state block)
NEW_TABS = '''
    # ──────────────────────────────────────────────────────────────────────────
    # NEW TAB: ACTIONS & PREDICTIONS
    # ──────────────────────────────────────────────────────────────────────────
    with tab_act:
        st.markdown("### 🎯 What to do next")
        for w in res.get("warnings", []):
            st.warning(w)
        recs = res.get("recommendations", [])
        if recs:
            for r in recs:
                st.markdown(f"**{r['priority']}. {r['title']}**")
                st.write(r["action"])
                st.caption(f"Expected impact: {r['impact']}")
        else:
            st.info("No specific recommendations - there is not enough signal in this data.")

        st.divider()
        st.markdown("### 📋 Your data with predictions, segments and flags")
        prev = res.get("preview", [])
        if prev:
            st.dataframe(pd.DataFrame(prev), use_container_width=True)
        ef = res.get("enriched_file")
        if ef and os.path.exists(ef):
            with open(ef, "rb") as fh:
                st.download_button("⬇️ Download full results (CSV)", fh.read(),
                                   file_name="AnchorIQ_results.csv", mime="text/csv", type="primary")
        elif res.get("session_id"):
            st.caption(f"Download the full results from: {api_url}/download/{res['session_id']}")

        fc = res.get("forecast")
        if fc:
            st.divider()
            st.markdown(f"### 🔮 Forecast: {fc['metric']}")
            st.write(", ".join(f"{x}: {y:,.0f}" for x, y in zip(fc["forecast"]["x"], fc["forecast"]["y"])))

    # ──────────────────────────────────────────────────────────────────────────
    # NEW TAB: INTERACTIVE CHARTS (Plotly)
    # ──────────────────────────────────────────────────────────────────────────
    with tab_viz:
        pcharts = res.get("plotly_charts", {})
        if not pcharts:
            st.info("Interactive charts need Plotly:  pip install plotly")
        else:
            import plotly.io as pio
            for _key, _ch in pcharts.items():
                st.plotly_chart(pio.from_json(_ch["json"]), use_container_width=True)

'''
swap("# ── Empty State ───", NEW_TABS.lstrip("\n") + "# ── Empty State ───")

swap("Supports CSV, Excel (.xlsx/.xls), and JSON files",
     "Supports CSV, Excel, JSON, JSONL, Parquet, XML, SQLite and PDF tables")

ast.parse(src)  # refuse to write a broken file
shutil.copy(PATH, PATH + ".bak")
open(PATH, "w", encoding="utf-8").write(src)
print("streamlit_app.py patched. Backup saved as streamlit_app.py.bak")
