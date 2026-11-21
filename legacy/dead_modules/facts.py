"""
services/facts.py
Turns model + data into CONCRETE, number-backed findings and recommendations.
The LLM later only *writes up* these facts - it never has to invent numbers.
"""
import numpy as np
import pandas as pd

DOMAIN = {
    "churn":   {"unit": "customers", "event": "churn",      "action": "a targeted retention offer or outreach"},
    "fraud":   {"unit": "transactions", "event": "fraud",   "action": "tighter rules or manual review"},
    "hr":      {"unit": "employees", "event": "attrition",   "action": "stay-interviews and a retention plan"},
    "sales":   {"unit": "records", "event": "sales",        "action": "focused sales and pricing effort"},
    "general": {"unit": "records", "event": "the outcome",  "action": "a focused intervention"},
}


def build_facts(df, info, model, scored, seg, anom, fc, analysis_type="general", filename=""):
    dom = DOMAIN.get(analysis_type, DOMAIN["general"])
    if analysis_type == "auto":
        dom = DOMAIN["general"]
    facts = {
        "overview": {"file": filename, "rows": int(len(df)), "columns": int(df.shape[1]),
                     "plan": info["plan"], "target": info["target"],
                     "target_confidence": info["target_confidence"],
                     "problem_type": info["problem_type"]},
        "findings": [], "recommendations": [], "warnings": [],
    }
    F, R, W = facts["findings"], facts["recommendations"], facts["warnings"]

    if info["target"] and info["target_confidence"] < 0.5:
        W.append(f"Target '{info['target']}' was auto-detected with low confidence - please confirm it is the outcome you care about.")

    if model and "error" not in model:
        _model_quality(model, F, W)
        if info["problem_type"] == "classification" and scored is not None and "prediction_score" in scored:
            _supervised_classification(df, info, model, scored, dom, F, R)
        elif info["problem_type"] == "regression":
            _supervised_regression(info, model, scored, dom, F, R)
        if model.get("outlier_note"):
            W.append(model["outlier_note"])
        if model.get("dropped_for_leakage"):
            W.append("Columns removed because they leak the answer: " +
                     ", ".join(f"{k} ({v})" for k, v in model["dropped_for_leakage"].items()))
    elif model and "error" in model:
        W.append(f"Prediction skipped: {model['error']}")

    if seg:
        F.append({"kind": "segments", "importance": 0.5,
                  "text": f"Records fall into {seg['k']} natural groups (silhouette {seg['silhouette']}).",
                  "evidence": seg["clusters"]})
        for c in sorted(seg["clusters"], key=lambda c: -c["size"])[:3]:
            R.append({"title": f"Treat segment {c['id'] + 1} differently ({c['share']:.0%} of records)",
                      "action": "Build a tailored offer/strategy for this group. Defining traits: " + "; ".join(c["traits"]),
                      "impact": "Segment-specific actions usually outperform one-size-fits-all.", "score": c["share"]})
    if anom and anom["count"]:
        F.append({"kind": "anomalies", "importance": 0.6,
                  "text": f"{anom['count']} records ({anom['share']:.1%}) look statistically unusual"
                          + (f", including {anom['extreme_count']} extreme value(s)." if anom.get("extreme_count") else "."),
                  "evidence": anom["top"][:5]})
        R.append({"title": "Review the flagged unusual records",
                  "action": f"Check the top {min(10, anom['count'])} outliers (e.g. {anom['top'][0]['reason']}) for data errors or fraud/opportunity.",
                  "impact": "Catches errors and exceptional cases early.", "score": 0.4})
    if fc:
        _forecast_facts(fc, F, R, W)

    if not info["target"]:
        _descriptive(df, info, F, W)

    F.sort(key=lambda f: -f["importance"])
    for i, r in enumerate(sorted(R, key=lambda r: -r.get("score", 0)), 1):
        r["priority"] = i
    facts["recommendations"] = sorted(R, key=lambda r: r["priority"])[:6]
    facts["findings"] = F[:10]
    return facts


# ── supervised ────────────────────────────────────────────────────────────
def _model_quality(model, F, W):
    if model["problem_type"] == "classification":
        if model.get("oof_recall") is not None:
            txt = (f"Best model: {model['best_model']} (AUC {model.get('auc')}). Flagging records it rates as risky catches "
                   f"{model['oof_recall']:.0%} of real '{model.get('positive_label')}' cases, and {model['oof_precision']:.0%} of its flags are correct "
                   f"(tested on data it had not seen).")
        else:
            txt = f"Best model: {model['best_model']} - AUC {model.get('auc')}, accuracy {model['accuracy']}%."
        weak = model.get("auc", 1) < 0.6
    else:
        txt = (f"Best model: {model['best_model']} - R2 {model['r2_score']}, typical error {model['mae']} "
               f"(guessing the average gives RMSE {model['baseline_rmse']} vs {model['rmse']} for the model).")
        weak = model["r2_score"] < 0.2
    F.append({"kind": "model", "text": txt, "importance": 0.7, "evidence": {}})
    if weak:
        W.append("The model finds only weak patterns in this data - treat predictions as indicative, and consider adding more informative columns.")


def _supervised_classification(df, info, model, scored, dom, F, R):
    target, pos = info["target"], model.get("positive_label")
    is_bin = pos is not None
    n = len(scored)
    if is_bin:
        y = (scored[target].astype(str) == str(pos)).astype(int)
        overall = float(y.mean())
        F.append({"kind": "base_rate", "importance": 0.9,
                  "text": f"Overall {target} = '{pos}' rate is {overall:.1%} ({int(y.sum()):,} of {n:,}).",
                  "evidence": {"rate": overall}})
        # gains
        order = np.argsort(-scored["prediction_score"].values)
        k = max(1, int(n * 0.1))
        cap = y.values[order][:k].sum() / max(y.sum(), 1)
        F.append({"kind": "gains", "importance": 0.95,
                  "text": f"The 10% highest-risk records ({k:,}) contain {cap:.0%} of all '{pos}' cases - {cap / 0.1:.1f}x better than picking at random.",
                  "evidence": {"top10_capture": cap}})
        R.insert(0, {"title": f"Act first on the {k:,} highest-risk {dom['unit']}",
                     "action": f"Use the downloadable predictions file (sorted by prediction_score) and apply {dom['action']} to the top 10%.",
                     "impact": f"Reaches {cap:.0%} of all {target} cases while touching only 10% of {dom['unit']}.", "score": 10})
    else:
        overall = None
    segs = _driver_segments(scored, model["feature_importance"], y if is_bin else None, overall, n, is_bin=True)
    _segment_findings(segs, target, pos, overall, dom, F, R, n, kind="rate")


def _supervised_regression(info, model, scored, dom, F, R):
    target = info["target"]
    y = pd.to_numeric(scored[target], errors="coerce")
    n, overall = len(scored), float(y.mean())
    F.append({"kind": "base_rate", "importance": 0.8,
              "text": f"Average {target} is {overall:,.2f} (median {y.median():,.2f}).", "evidence": {}})
    if (y.dropna() >= 0).all() and y.sum() > 0:
        top = y.sort_values(ascending=False).head(max(1, int(n * 0.2))).sum() / y.sum()
        F.append({"kind": "pareto", "importance": 0.85,
                  "text": f"The top 20% of records generate {top:.0%} of total {target}.", "evidence": {"top20_share": top}})
        if top > 0.5:
            R.append({"title": f"Protect and grow the top 20% that drive {top:.0%} of {target}",
                      "action": "Identify these records (sort by the target) and give them priority service, retention and upsell attention.",
                      "impact": "Small group, outsized effect on the total.", "score": 8})
    segs = _driver_segments(scored, model["feature_importance"], y, overall, n, is_bin=False)
    _segment_findings(segs, target, None, overall, dom, F, R, n, kind="mean")


def bin_labels(s: pd.Series, q: int = 4) -> pd.Series:
    """Quartile labels like '1 to 9', '9 to 29' (first bin starts at the true minimum)."""
    bins, edges = pd.qcut(s, q, duplicates="drop", retbins=True)
    edges = list(edges)
    edges[0] = float(s.min())
    names = {iv: f"{edges[i]:,.4g} to {edges[i + 1]:,.4g}" for i, iv in enumerate(bins.cat.categories)}
    return bins.map(names).astype(str)


def _driver_segments(frame, importance, y, overall, n, is_bin):
    out = []
    min_n = max(30, int(0.02 * n))
    for col in list(importance)[:6]:
        if col not in frame.columns:
            continue
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            try:
                labels = bin_labels(s)
            except Exception:
                continue
        else:
            labels = s.astype(str)
        g = pd.DataFrame({"seg": labels.values, "y": np.asarray(y)}).groupby("seg")["y"].agg(["mean", "count"])
        for seg, row in g.iterrows():
            if row["count"] >= min_n:
                out.append({"column": col, "segment": seg, "n": int(row["count"]), "share": row["count"] / n,
                            "value": float(row["mean"]), "lift": float(row["mean"] / overall) if overall else None,
                            "importance": float(importance[col])})
    return out


def _segment_findings(segs, target, pos, overall, dom, F, R, n, kind):
    if not segs or not overall:
        return
    high = sorted([s for s in segs if s["lift"] and s["lift"] > 1.15], key=lambda s: -(s["n"] * (s["value"] - overall)))
    low = sorted([s for s in segs if s["lift"] and s["lift"] < 0.85], key=lambda s: s["value"])
    fmt = (lambda v: f"{v:.1%}") if kind == "rate" else (lambda v: f"{v:,.2f}")
    what = f"'{pos}' rate" if kind == "rate" else f"average {target}"
    seen = set()
    for s in high[:6]:
        if s["column"] in seen:
            continue
        seen.add(s["column"])
        F.append({"kind": "driver", "importance": 0.6 + s["importance"],
                  "text": f"{s['column']} = {s['segment']}: {what} is {fmt(s['value'])} vs {fmt(overall)} overall "
                          f"({s['lift']:.1f}x) across {s['n']:,} records ({s['share']:.0%}).",
                  "evidence": s})
        if kind == "rate":
            saved = s["n"] * s["value"] * 0.10
            R.append({"title": f"Focus on {s['column']} = {s['segment']}",
                      "action": f"Apply {dom['action']} to this group ({s['n']:,} {dom['unit']}, {fmt(s['value'])} {target} rate).",
                      "impact": f"Assumption: if the rate in this group fell by 10% (relative), about {saved:,.0f} fewer {target} cases.",
                      "score": s["n"] * (s["value"] - overall) / n * 10})
        else:
            R.append({"title": f"Grow {s['column']} = {s['segment']}",
                      "action": f"This group averages {fmt(s['value'])} {target} vs {fmt(overall)} overall - acquire/serve more of it.",
                      "impact": f"{s['share']:.0%} of records, {s['lift']:.1f}x the average value.",
                      "score": s["share"] * (s["lift"] - 1) * 5})
    for s in low[:2]:
        F.append({"kind": "protective", "importance": 0.5 + s["importance"],
                  "text": f"{s['column']} = {s['segment']}: {what} is only {fmt(s['value'])} vs {fmt(overall)} overall ({s['n']:,} records).",
                  "evidence": s})
    if low and kind == "rate":
        s = low[0]
        R.append({"title": f"Learn from {s['column']} = {s['segment']}",
                  "action": f"This group has the lowest {target} rate ({fmt(s['value'])}). Find out what is different and move other {dom['unit']} towards it.",
                  "impact": f"Closing even part of the gap to {fmt(overall)} compounds across the base.", "score": 0.3})


# ── descriptive / time series ────────────────────────────────────────────
def _forecast_facts(fc, F, R, W):
    f = fc["forecast"]["y"]
    last = fc["history"]["y"][-1]
    txt = (f"Trend: {fc['metric']} is changing by {fc['trend_per_period']:+,.2f} per {fc['freq']}"
           + (f"; latest {fc['freq']}s are {fc['growth_pct']:+.1f}% vs the prior period" if fc["growth_pct"] is not None else "") + ".")
    F.append({"kind": "trend", "importance": 0.85, "text": txt, "evidence": {"growth": fc["growth_pct"]}})
    F.append({"kind": "forecast", "importance": 0.8,
              "text": f"Forecast for the next {len(f)} {fc['freq']}(s): {', '.join(f'{v:,.0f}' for v in f)} "
                      f"(last observed {last:,.0f}).", "evidence": fc["backtest"]})
    bt = fc["backtest"]
    if fc.get("outliers_capped"):
        W.append(f"{fc['outliers_capped']} extreme value(s) of {fc['metric']} were capped before forecasting so they would not distort the trend.")
    if bt["beats_naive"]:
        F.append({"kind": "forecast_quality", "importance": 0.4,
                  "text": f"Back-test: forecast error {bt['mape_pct']}% (MAPE), better than the 'same as last period' baseline.", "evidence": bt})
    else:
        W.append("The forecast did not beat a simple 'same as last period' baseline in back-testing - treat it as a rough indication only.")
    if fc.get("peak") is not None and bt["beats_naive"] and (fc.get("peak_lift_pct") or 0) >= 10:
        F.append({"kind": "seasonality", "importance": 0.5,
                  "text": f"Strongest period in the cycle: {fc['peak']} ({fc['peak_lift_pct']:.0f}% above the average).", "evidence": {}})
        R.append({"title": f"Plan staffing/stock/marketing around {fc['peak']}",
                  "action": f"{fc['peak']} runs {fc['peak_lift_pct']:.0f}% above average - schedule capacity and campaigns for it.",
                  "impact": "Avoids lost sales in busy periods and idle cost in quiet ones.", "score": 2})
    if fc["growth_pct"] is not None and fc["growth_pct"] < -5:
        R.append({"title": f"Investigate the {abs(fc['growth_pct']):.0f}% recent decline in {fc['metric']}",
                  "action": "Break the metric down by product/region/channel to find where the drop comes from.",
                  "impact": "Stopping a decline early is cheaper than recovering later.", "score": 6})
    elif fc["growth_pct"] is not None and fc["growth_pct"] > 5:
        R.append({"title": f"Double down on what is working (+{fc['growth_pct']:.0f}% recently)",
                  "action": "Find which segment/channel drives the growth and increase investment there.",
                  "impact": "Momentum is the cheapest growth.", "score": 5})


def _descriptive(df, info, F, W):
    nums = info["numeric_cols"]
    if len(nums) >= 2:
        c = df[nums[:15]].corr().abs()
        pairs = [(c.iloc[i, j], nums[i], nums[j]) for i in range(len(c)) for j in range(i + 1, len(c))]
        pairs.sort(reverse=True)
        for r, a, b in pairs[:2]:
            if r > 0.5:
                F.append({"kind": "correlation", "importance": 0.55,
                          "text": f"{a} and {b} move together (correlation {df[a].corr(df[b]):+.2f}).", "evidence": {}})
    miss = df.isna().mean()
    bad = miss[miss > 0.2]
    if len(bad):
        W.append("High share of missing values in: " + ", ".join(f"{k} ({v:.0%})" for k, v in bad.items()))
