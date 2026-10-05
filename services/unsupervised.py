"""
services/unsupervised.py
For data WITHOUT a clear target: customer/record segmentation (KMeans),
anomaly detection (IsolationForest). Plus time-series trend + forecast.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def _matrix(df, info, exclude=()):
    num = [c for c in info["numeric_cols"] if c not in exclude]
    cat = [c for c in info["categorical_cols"] if c not in exclude and df[c].nunique() <= 15]
    if not num and not cat:
        return None, num, cat
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype(str)
    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                            min_frequency=0.02))]), cat),
    ])
    return pre.fit_transform(X), num, cat


def segmentation(df: pd.DataFrame, info: dict, max_k: int = 6):
    M, num, cat = _matrix(df, info)
    if M is None or len(df) < 50:
        return None
    sample = np.random.RandomState(0).choice(len(M), min(4000, len(M)), replace=False)
    best_k, best_s = 2, -1
    for k in range(2, max_k + 1):
        km = KMeans(k, n_init=5, random_state=0).fit(M[sample])
        s = silhouette_score(M[sample], km.labels_)
        if s > best_s:
            best_k, best_s = k, s
    km = KMeans(best_k, n_init=10, random_state=0).fit(M)
    labels = km.labels_

    clusters = []
    for k in range(best_k):
        mask = labels == k
        feats = []
        for c in num:
            sd = df[c].std() or 1
            z = (df.loc[mask, c].mean() - df[c].mean()) / sd
            feats.append((abs(z), f"{c} {'higher' if z > 0 else 'lower'} than average "
                                  f"({df.loc[mask, c].mean():,.2f} vs {df[c].mean():,.2f})"))
        for c in cat:
            top = df.loc[mask, c].astype(str).value_counts(normalize=True)
            base = df[c].astype(str).value_counts(normalize=True)
            v = top.index[0]
            diff = top.iloc[0] - base.get(v, 0)
            feats.append((abs(diff) * 3, f"{c} = {v} ({top.iloc[0]:.0%} vs {base.get(v, 0):.0%} overall)"))
        feats.sort(reverse=True)
        clusters.append({"id": k, "size": int(mask.sum()), "share": round(float(mask.mean()), 3),
                         "traits": [t for _, t in feats[:3]]})
    xy = PCA(2, random_state=0).fit_transform(M)
    pts = np.random.RandomState(1).choice(len(M), min(2000, len(M)), replace=False)
    return {"k": best_k, "silhouette": round(float(best_s), 3), "labels": labels.tolist(),
            "clusters": clusters,
            "pca": {"x": xy[pts, 0].round(3).tolist(), "y": xy[pts, 1].round(3).tolist(),
                    "cluster": labels[pts].tolist()}}


def anomalies(df: pd.DataFrame, info: dict, top_n: int = 10):
    num = info["numeric_cols"]
    if len(num) < 1 or len(df) < 50:
        return None
    X = df[num].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    # 1) explainable single-column extremes (robust z-score, resistant to the outliers themselves)
    med = X.median()
    mad = (X - med).abs().median().replace(0, np.nan)
    sd = X.std().replace(0, 1)
    rz = ((X - med) / (1.4826 * mad)).fillna((X - med) / sd)
    zmax = rz.abs().max(axis=1).values
    extreme = zmax > 8
    # 2) multivariate oddities
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(X)
    score = -iso.score_samples(X)
    flag = (iso.predict(X) == -1) | extreme
    order = np.lexsort((-score, -extreme.astype(int)))  # extremes first, then by isolation score
    rows = []
    for i in order[:top_n]:
        col = rz.iloc[i].abs().idxmax()
        rows.append({"row": int(df.index[i]), "score": round(float(score[i]), 3),
                     "reason": f"{col} = {X.iloc[i][col]:,.2f} (typical is about {med[col]:,.2f})"})
    return {"count": int(flag.sum()), "share": round(float(flag.mean()), 4),
            "extreme_count": int(extreme.sum()),
            "flags": flag.tolist(), "scores": score.round(4).tolist(), "top": rows}


def forecast(df: pd.DataFrame, time_col: str, metric: str, horizon: int | None = None):
    d = df[[time_col, metric]].dropna()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna()
    if len(d) < 20:
        return None
    span_days = (d[time_col].max() - d[time_col].min()).days
    if span_days < 14:
        return None
    if span_days <= 120:
        freq, season, label = "D", 7, "day"
    elif span_days <= 800:
        freq, season, label = "W", 52 if span_days > 730 else 0, "week"
    else:
        freq, season, label = "MS", 12, "month"
    # one absurd value must not create a fake spike/trend in a period total
    hi_cap, lo_cap = d[metric].quantile(0.995), d[metric].quantile(0.005)
    capped = int(((d[metric] > hi_cap) | (d[metric] < lo_cap)).sum())
    if capped and (d[metric].max() > 3 * max(abs(hi_cap), 1e-9) or d[metric].min() < 3 * min(lo_cap, -1e-9) * -1 and False):
        d[metric] = d[metric].clip(lo_cap, hi_cap)
    else:
        capped = 0
    agg = "mean" if any(k in metric.lower() for k in ("price", "rate", "score", "rating", "avg")) else "sum"
    pcode = {"D": "D", "W": "W", "MS": "M"}[freq]
    d["_p"] = d[time_col].dt.to_period(pcode)
    g = d.groupby("_p")[metric].agg(agg)
    full = pd.period_range(g.index.min(), g.index.max(), freq=pcode)
    g = g.reindex(full)
    g = g.fillna(0) if agg == "sum" else g.interpolate()
    if freq != "D":  # a half-finished first/last period would look like a fake rise/drop
        if d[time_col].max().normalize() < full[-1].end_time.normalize():
            g = g.iloc[:-1]
        if len(g) and d[time_col].min().normalize() > full[0].start_time.normalize():
            g = g.iloc[1:]
    s = pd.Series(g.values.astype(float), index=g.index.to_timestamp())
    n = len(s)
    if n < 8:
        return None
    if season and n < 2 * season:
        season = 0
    t = np.arange(n)

    def feats(tt):
        cols = [tt.reshape(-1, 1)]
        if season:
            cols.append(np.eye(season)[tt % season])
        return np.hstack(cols)

    h = horizon or (7 if freq == "D" else 4 if freq == "W" else 3)
    test_n = min(max(3, n // 5), 12)
    tr = slice(0, n - test_n)
    model = Ridge(alpha=1.0).fit(feats(t[tr]), s.values[tr])
    bt = model.predict(feats(t[n - test_n:]))
    actual = s.values[n - test_n:]
    mae = float(np.mean(np.abs(bt - actual)))
    naive = float(np.mean(np.abs(s.values[n - test_n - 1] - actual)))
    mape = float(np.mean(np.abs(bt - actual) / np.maximum(np.abs(actual), 1e-9)) * 100)

    model = Ridge(alpha=1.0).fit(feats(t), s.values)
    resid = s.values - model.predict(feats(t))
    sd = float(np.std(resid))
    tf = np.arange(n, n + h)
    fc = model.predict(feats(tf))
    idx_f = pd.date_range(s.index[-1], periods=h + 1, freq={"D": "D", "W": "7D", "MS": "MS"}[freq])[1:]

    k = max(1, min(3, n // 4))
    recent, prior = s.values[-k:].sum(), s.values[-2 * k:-k].sum()
    growth = (recent - prior) / abs(prior) * 100 if prior else None
    slope = float(np.polyfit(t, s.values, 1)[0])

    peak, peak_lift = None, None
    if season and n >= 3 * season and freq == "D":
        dow = s.groupby(s.index.dayofweek).mean()
        peak = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(dow.idxmax())]
        peak_lift = (dow.max() - dow.mean()) / abs(dow.mean()) * 100 if dow.mean() else None
    elif season and n >= 3 * season and freq == "MS":
        import calendar
        mo = s.groupby(s.index.month).mean()
        peak = calendar.month_name[int(mo.idxmax())]
        peak_lift = (mo.max() - mo.mean()) / abs(mo.mean()) * 100 if mo.mean() else None

    return {
        "metric": metric, "freq": label, "aggregation": agg, "periods": n, "outliers_capped": capped,
        "history": {"x": [str(i.date()) for i in s.index], "y": s.round(2).tolist()},
        "forecast": {"x": [str(i.date()) for i in idx_f], "y": fc.round(2).tolist(),
                     "lower": (fc - 1.96 * sd).round(2).tolist(), "upper": (fc + 1.96 * sd).round(2).tolist()},
        "growth_pct": None if growth is None else round(float(growth), 1),
        "trend_per_period": round(slope, 3), "peak": peak,
        "peak_lift_pct": None if peak_lift is None else round(float(peak_lift), 1),
        "backtest": {"mae": round(mae, 2), "naive_mae": round(naive, 2), "mape_pct": round(mape, 1),
                     "beats_naive": bool(mae < naive)},
    }
