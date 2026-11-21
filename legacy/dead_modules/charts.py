"""
services/charts.py
Interactive Plotly charts. Each function returns a Plotly JSON string
(render in Streamlit with:  st.plotly_chart(plotly.io.from_json(js), use_container_width=True)).
Requires: pip install plotly
"""
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

PALETTE = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#3B82F6", "#8B5CF6"]


def _j(fig, title):
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=40, r=20, t=60, b=40),
                      colorway=PALETTE, height=380)
    return fig.to_json()


def build_charts(df, info, model, scored, seg, fc, facts):
    """Return {chart_key: {"title":..., "json":...}}"""
    if not HAS_PLOTLY:
        return {}
    out = {}

    def add(key, title, fn):
        try:
            fig = fn()
            if fig is not None:
                out[key] = {"title": title, "json": _j(fig, title)}
        except Exception as e:
            print(f"[WARN] chart {key} failed: {e}")

    t = info["target"]
    nums = info["numeric_cols"]

    if t:
        def target_dist():
            s = df[t]
            if info["problem_type"] == "classification":
                vc = s.astype(str).value_counts()
                return go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=PALETTE[0],
                                        text=vc.values, textposition="outside"))
            return px.histogram(s, nbins=40)
        add("target", f"Distribution of {t}", target_dist)

    if model and model.get("feature_importance"):
        def importance():
            items = sorted(model["feature_importance"].items(), key=lambda kv: kv[1])
            return go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                                    marker_color=PALETTE[0]))
        add("drivers", "What drives the outcome (share of predictive power)", importance)

    drivers = [f for f in facts["findings"] if f["kind"] == "driver"][:1]
    if drivers and scored is not None and info["problem_type"] == "classification":
        col = drivers[0]["evidence"]["column"]

        def seg_rate():
            pos = model.get("positive_label")
            y = (scored[t].astype(str) == str(pos)).astype(int)
            s = scored[col]
            from services.facts import bin_labels
            lab = (bin_labels(s) if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8 else s.astype(str))
            g = y.groupby(lab.values).agg(["mean", "count"]).sort_values("mean", ascending=False)
            fig = go.Figure(go.Bar(x=g.index, y=g["mean"] * 100, marker_color=PALETTE[3],
                                   text=[f"{v:.0f}%" for v in g["mean"] * 100], textposition="outside"))
            fig.add_hline(y=float(y.mean()) * 100, line_dash="dash", annotation_text="overall")
            fig.update_yaxes(title=f"{t} rate %")
            return fig
        add("segment_rate", f"{t} rate by {col}", seg_rate)

    if scored is not None and "prediction_score" in scored and info["problem_type"] == "classification" \
            and model.get("positive_label") is not None:
        def gains():
            y = (scored[t].astype(str) == str(model["positive_label"])).astype(int).values
            order = np.argsort(-scored["prediction_score"].values)
            cum = np.cumsum(y[order]) / max(y.sum(), 1)
            x = np.arange(1, len(y) + 1) / len(y)
            fig = go.Figure()
            fig.add_scatter(x=x * 100, y=cum * 100, name="Model", line=dict(color=PALETTE[0], width=3))
            fig.add_scatter(x=[0, 100], y=[0, 100], name="Random", line=dict(dash="dash", color="#94a3b8"))
            fig.update_xaxes(title="% of records contacted (highest risk first)")
            fig.update_yaxes(title="% of all cases captured")
            return fig
        add("gains", "How much a targeted approach captures", gains)

        def tiers():
            vc = scored["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
            return go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=["#EF4444", "#F59E0B", "#10B981"]))
        add("risk_tiers", "Records by predicted risk tier", tiers)

    if len(nums) >= 2:
        def corr():
            c = df[nums[:12]].corr()
            return go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.columns, zmin=-1, zmax=1,
                                        colorscale="RdBu", reversescale=True,
                                        text=np.round(c.values, 2), texttemplate="%{text}"))
        add("correlation", "Correlation between numeric columns", corr)

    if nums:
        def dist():
            cols = nums[:6]
            fig = make_subplots(rows=2, cols=3, subplot_titles=cols)
            for i, c in enumerate(cols):
                fig.add_histogram(x=df[c].dropna(), row=i // 3 + 1, col=i % 3 + 1, showlegend=False,
                                  marker_color=PALETTE[0])
            return fig
        add("distributions", "Distributions of key numeric columns", dist)

    if fc:
        def forecast_fig():
            fig = go.Figure()
            fig.add_scatter(x=fc["history"]["x"], y=fc["history"]["y"], name="Actual", line=dict(color=PALETTE[0]))
            fx = fc["forecast"]["x"]
            fig.add_scatter(x=fx + fx[::-1], y=fc["forecast"]["upper"] + fc["forecast"]["lower"][::-1],
                            fill="toself", fillcolor="rgba(99,102,241,0.15)", line=dict(width=0),
                            name="Likely range", hoverinfo="skip")
            fig.add_scatter(x=fx, y=fc["forecast"]["y"], name="Forecast", line=dict(color=PALETTE[3], dash="dash"))
            fig.update_yaxes(title=f"{fc['metric']} per {fc['freq']}")
            return fig
        add("forecast", f"{fc['metric']}: history and forecast", forecast_fig)

    if seg:
        def clusters():
            p = seg["pca"]
            return px.scatter(x=p["x"], y=p["y"], color=[f"Segment {c + 1}" for c in p["cluster"]],
                              labels={"x": "pattern axis 1", "y": "pattern axis 2", "color": ""})
        add("segments", "Natural segments in your data", clusters)
    return out
