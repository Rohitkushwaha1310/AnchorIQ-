"""
services/pipeline.py
ONE orchestration function used by BOTH the FastAPI backend and the Streamlit app.
load -> profile -> clean -> (EDA) -> model / segments / anomalies / forecast -> facts -> insights
"""
import math
import os
import uuid

import numpy as np
import pandas as pd

from services.charts import build_charts
from services.cleaning import auto_clean
from services.eda import auto_eda
from services.facts import build_facts
from services.inspection import inspect_dataset
from services.insights import generate_insights
from services.modeling import auto_model
from services.profiler import profile
from services.unsupervised import anomalies, forecast, segmentation


def run_analysis(df_raw: pd.DataFrame, filename: str = "data", analysis_type: str = "auto",
                 target_column: str | None = None, session_id: str | None = None,
                 load_notes: list | None = None, base_dir: str = ".") -> dict:
    session_id = session_id or uuid.uuid4().hex[:8]
    charts_dir = os.path.join(base_dir, "charts", session_id)
    reports_dir = os.path.join(base_dir, "reports")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(reports_dir, exist_ok=True)

    # 1 ── understand the raw data, find the target, drop unlabeled rows
    _, first = profile(df_raw, analysis_type, target_column)
    target = first["target"]
    inspection = inspect_dataset(df_raw, analysis_type=analysis_type, target_column=target)
    work = df_raw
    if target:
        work = df_raw[df_raw[target].notna()]

    # 2 ── clean, then profile the cleaned frame
    df_clean, clean_report = auto_clean(work.copy())
    df_clean, info = profile(df_clean, analysis_type, target)
    info["target_confidence"], info["target_reason"] = first["target_confidence"], first["target_reason"]

    # 3 ── analyses
    model, scored = ({}, None)
    if "supervised" in info["plan"]:
        model, scored = auto_model(df_clean, info, save_path=os.path.join(models_dir, f"{session_id}.pkl"))
    seg = segmentation(df_clean, info) if "segmentation" in info["plan"] else None
    anom = anomalies(df_clean, info) if "anomalies" in info["plan"] else None
    fc = None
    if "forecast" in info["plan"]:
        try:
            fc = forecast(df_clean, info["time_col"], info["metric"])
        except Exception as e:
            print(f"[WARN] forecast failed: {e}")

    # 4 ── facts -> insights
    facts = build_facts(df_clean, info, model, scored, seg, anom, fc, analysis_type, filename)
    insights = generate_insights(facts, analysis_type)

    # 5 ── charts: interactive (UI) + static PNG (PDF report)
    plotly_charts = build_charts(df_clean, info, model, scored, seg, fc, facts)
    eda_df = df_clean.copy()
    if info["problem_type"] == "classification" and model.get("positive_label") is not None:
        eda_df[target] = (eda_df[target].astype(str) == str(model["positive_label"])).astype(int)
    try:
        chart_paths = auto_eda(eda_df, target=target, analysis_type=analysis_type, save_dir=charts_dir)
    except Exception as e:
        print(f"[WARN] PNG charts failed: {e}")
        chart_paths = []

    # 6 ── enriched dataset for download
    out = df_clean.copy()
    if scored is not None:
        extra = [c for c in scored.columns if c not in out.columns]
        out = out.join(scored[extra])
        if "prediction_score" in out:
            out = out.sort_values("prediction_score", ascending=False)
    if seg:
        out["segment"] = [f"Segment {l + 1}" for l in seg["labels"]]
    if anom:
        out["is_anomaly"] = anom["flags"]
        out["anomaly_score"] = anom["scores"]
    pred_path = os.path.join(reports_dir, f"{session_id}_enriched.csv")
    out.to_csv(pred_path, index=False)

    seg_public = None if not seg else {k: v for k, v in seg.items() if k != "labels"}
    anom_public = None if not anom else {k: v for k, v in anom.items() if k not in ("flags", "scores")}

    result = {
        "filename": filename, "session_id": session_id, "analysis_type": analysis_type,
        "load_notes": load_notes or [],
        "profile": {k: info[k] for k in ("target", "target_confidence", "target_reason", "problem_type",
                                         "positive_label", "time_col", "metric", "plan", "column_types")},
        "inspection": inspection, "cleaning": clean_report,
        "model_results": model, "segments": seg_public, "anomalies": anom_public, "forecast": fc,
        "facts": facts, "recommendations": facts["recommendations"], "warnings": facts["warnings"],
        "insights": insights, "plotly_charts": plotly_charts, "chart_paths": chart_paths,
        "enriched_file": pred_path,
        "preview": json_safe(out.head(50).to_dict("records")),
        "status": "Analysis Complete!",
    }
    return json_safe(result)


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    return o
