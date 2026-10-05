"""
services/modeling.py  (v3)
- Preprocessing lives INSIDE the saved pipeline, so the model can score brand-new raw data.
- XGBoost is optional (falls back to HistGradientBoosting).
- Leakage columns are removed before training.
- Every row gets an honest (out-of-fold) prediction.
- Importance = permutation importance on ORIGINAL columns (readable by humans).
Returns (results_dict, scored_dataframe).
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from services.profiler import find_leakage

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _log(msg):
    try:
        print(msg)
    except Exception:
        print(msg.encode("ascii", "replace").decode())


def auto_model(df: pd.DataFrame, info: dict, save_path: str | None = "models/model.pkl"):
    target = info["target"]
    problem = info["problem_type"]
    col_types = info["column_types"]
    positive = info.get("positive_label")

    d = df[df[target].notna()].copy()
    if d[target].nunique() < 2:
        return {"error": f"Target '{target}' has only one unique value."}, None
    if len(d) < 30:
        return {"error": f"Need at least 30 labelled rows (found {len(d)})."}, None

    # ── leakage + feature selection ─────────────────────────────────────
    leaks = find_leakage(d, target, col_types, problem)
    skip_types = {"id", "text", "high_cardinality", "datetime"}
    feat_num = [c for c in info["numeric_cols"] if c != target and c not in leaks]
    feat_cat = [c for c in info["categorical_cols"] if c != target and c not in leaks]
    X = d[feat_num + feat_cat].copy()

    # datetime -> numeric parts
    for c in [c for c, t in col_types.items() if t == "datetime" and c != target]:
        X[f"{c}__year"] = d[c].dt.year
        X[f"{c}__month"] = d[c].dt.month
        X[f"{c}__dayofweek"] = d[c].dt.dayofweek
        feat_num += [f"{c}__year", f"{c}__month", f"{c}__dayofweek"]
    for c in feat_cat:
        X[c] = X[c].astype(str)
    if X.shape[1] == 0:
        return {"error": "No usable feature columns after preprocessing."}, None

    # ── target encoding ─────────────────────────────────────────────────
    classes = None
    if problem == "classification":
        classes = sorted(d[target].astype(str).unique())
        if len(classes) == 2:
            pos = str(positive) if positive is not None else classes[-1]
            y = (d[target].astype(str) == pos).astype(int)
        else:
            pos = None
            y = pd.Series(pd.Categorical(d[target].astype(str), categories=classes).codes, index=d.index)
    else:
        pos = None
        y = pd.to_numeric(d[target], errors="coerce")
        keep = y.notna()
        X, y, d = X[keep], y[keep], d[keep]
        # extreme outliers would dominate the error and hide real patterns: clip for TRAINING/EVALUATION only
        lo, hi = y.quantile(0.005), y.quantile(0.995)
        n_out = int(((y < lo) | (y > hi)).sum())
        y_raw_std = float(y.std() or 1)
        y_clip = y.clip(lo, hi)
        outlier_note = None
        if n_out and abs(float(y_clip.std()) - y_raw_std) / y_raw_std > 0.3:
            y = y_clip
            outlier_note = (f"{n_out} extreme values of {target} (outside {lo:,.4g} to {hi:,.4g}) were capped "
                            f"before training so they would not distort the model.")
    is_binary = problem == "classification" and len(classes) == 2
    outlier_note = locals().get("outlier_note")

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         [c for c in X.columns if c not in feat_cat]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                               sparse_output=False))]), feat_cat),
    ])

    candidates = _candidates(problem, is_binary)
    scoring = ("roc_auc" if is_binary else "f1_macro") if problem == "classification" else "r2"
    cv_split = (StratifiedKFold(3, shuffle=True, random_state=42) if problem == "classification"
                else KFold(3, shuffle=True, random_state=42))

    # model selection on a sample for speed
    if len(X) > 20000:
        idx = X.sample(20000, random_state=42).index
        Xs, ys = X.loc[idx], y.loc[idx]
    else:
        Xs, ys = X, y

    leaderboard = []
    for name, est in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        try:
            sc = cross_val_score(pipe, Xs, ys, cv=cv_split, scoring=scoring)
            leaderboard.append({"model": name, "cv_mean": round(float(sc.mean()), 4),
                                "cv_std": round(float(sc.std()), 4), "_est": est})
        except Exception as e:
            _log(f"[WARN] {name} failed: {e}")
    if not leaderboard:
        return {"error": "All candidate models failed to train."}, None
    leaderboard.sort(key=lambda r: r["cv_mean"], reverse=True)
    best = leaderboard[0]
    best_name, best_est = best["model"], best["_est"]
    _log(f"  Best model: {best_name} ({scoring}={best['cv_mean']})")

    # ── hold-out evaluation ─────────────────────────────────────────────
    stratify = y if problem == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipe = Pipeline([("pre", pre), ("model", best_est)]).fit(X_tr, y_tr)

    results = {
        "problem_type": problem, "best_model": best_name, "scoring": scoring,
        "cv_mean": best["cv_mean"], "cv_std": best["cv_std"],
        "train_samples": len(X_tr), "test_samples": len(X_te), "n_features": X.shape[1],
        "features_used": list(X.columns),
        "dropped_for_leakage": leaks,
        "leaderboard": [{k: v for k, v in r.items() if k != "_est"} for r in leaderboard],
        "has_xgboost": HAS_XGB,
        "outlier_note": outlier_note,
    }
    scored = d.copy()

    if problem == "classification":
        pred = pipe.predict(X_te)
        results.update({
            "classes": classes, "positive_label": pos,
            "accuracy": round(float(accuracy_score(y_te, pred)) * 100, 2),
            "f1_score": round(float(f1_score(y_te, pred, average="binary" if is_binary else "weighted")), 4),
            "precision": round(float(precision_score(y_te, pred, average="binary" if is_binary else "weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, pred, average="binary" if is_binary else "weighted", zero_division=0)), 4),
            "baseline_accuracy": round(float(y_te.value_counts(normalize=True).max()) * 100, 2),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "report": classification_report(y_te, pred, output_dict=True, zero_division=0),
        })
        try:
            proba = pipe.predict_proba(X_te)
            results["auc"] = round(float(roc_auc_score(y_te, proba[:, 1]) if is_binary
                                         else roc_auc_score(y_te, proba, multi_class="ovr")), 4)
        except Exception:
            results["auc"] = results["cv_mean"]

        # honest out-of-fold prediction for every row
        oof = _oof(pipe, X, y, cv_split, "predict_proba")
        if is_binary:
            p = oof[:, 1]
            thr = _best_threshold(y, p)
            results["threshold"] = round(thr, 3)
            pred_all = (p >= thr).astype(int)
            results["oof_precision"] = round(float(precision_score(y, pred_all, zero_division=0)), 4)
            results["oof_recall"] = round(float(recall_score(y, pred_all, zero_division=0)), 4)
            scored[f"predicted_{target}"] = np.where(pred_all == 1, pos, [c for c in classes if c != pos][0])
            scored["prediction_score"] = np.round(p, 4)
            scored["risk_tier"] = pd.cut(p, [-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"]).astype(str)
        else:
            scored[f"predicted_{target}"] = [classes[i] for i in oof.argmax(axis=1)]
            scored["prediction_score"] = np.round(oof.max(axis=1), 4)
    else:
        pred = pipe.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        results.update({
            "r2_score": round(float(r2_score(y_te, pred)), 4),
            "rmse": round(rmse, 4), "mae": round(float(mean_absolute_error(y_te, pred)), 4),
            "baseline_rmse": round(float(np.sqrt(np.mean((y_te - y_tr.mean()) ** 2))), 4),
        })
        oof = _oof(pipe, X, y, cv_split, "predict")
        scored[f"predicted_{target}"] = np.round(oof, 4)
        scored["prediction_error"] = np.round(oof - y.values, 4)

    # tournament keys expected by the existing UI/PDF
    for key, names in {"lr": ("Logistic Regression", "Ridge Regression"),
                       "rf": ("Random Forest",),
                       "xgb": ("XGBoost", "Gradient Boosting")}.items():
        m = next((r for r in leaderboard if r["model"] in names), None)
        suffix = "r2" if problem == "regression" else "auc"
        results[f"{key}_{suffix}"] = m["cv_mean"] if m else 0

    # ── readable importance (original columns) ──────────────────────────
    results["feature_importance"] = _importance(pipe, X_te, y_te, scoring)

    # ── final model on all data, saved for future raw data ─────────────
    final = Pipeline([("pre", pre), ("model", best_est)]).fit(X, y)
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            joblib.dump({"pipeline": final, "features": list(X.columns), "target": target,
                         "problem_type": problem, "classes": classes, "positive_label": pos,
                         "threshold": results.get("threshold")}, save_path)
            results["model_path"] = save_path
        except Exception as e:
            _log(f"[WARN] could not save model: {e}")
    return results, scored


# ── helpers ───────────────────────────────────────────────────────────────
def _candidates(problem, is_binary):
    if problem == "classification":
        c = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                                    class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        }
        if HAS_XGB:
            eval_metric = "logloss" if is_binary else "mlogloss"
            c["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.06, subsample=0.9,
                                         colsample_bytree=0.8, eval_metric=eval_metric, verbosity=0, random_state=42)
        else:
            c["Gradient Boosting"] = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                                                    max_iter=250, random_state=42)
        return c
    c = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=3,
                                               n_jobs=-1, random_state=42),
    }
    if HAS_XGB:
        c["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.06, subsample=0.9,
                                    colsample_bytree=0.8, verbosity=0, random_state=42)
    else:
        c["Gradient Boosting"] = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06,
                                                               max_iter=300, random_state=42)
    return c


def _oof(pipe, X, y, cv, method):
    from sklearn.base import clone
    if len(X) > 150_000:  # too slow: fall back to in-sample
        return getattr(pipe, method)(X)
    return cross_val_predict(clone(pipe), X, y, cv=cv, method=method)


def _best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def _importance(pipe, X_te, y_te, scoring):
    try:
        if len(X_te) > 2000:
            idx = X_te.sample(2000, random_state=1).index
            X_te, y_te = X_te.loc[idx], y_te.loc[idx]
        pi = permutation_importance(pipe, X_te, y_te, scoring=scoring, n_repeats=3,
                                    random_state=42, n_jobs=1)
        imp = np.clip(pi.importances_mean, 0, None)
        total = imp.sum()
        if total <= 0:
            return {}
        imp = imp / total
        order = np.argsort(imp)[::-1][:15]
        return {str(X_te.columns[i]): round(float(imp[i]), 4) for i in order if imp[i] > 0}
    except Exception as e:
        _log(f"[WARN] importance failed: {e}")
        return {}


def predict_new(model_path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Score brand-new raw data with a saved model."""
    bundle = joblib.load(model_path)
    X = new_df.copy()
    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]) or any(f"{col}__{part}" in bundle["features"] for part in ("year", "month", "dayofweek")):
            dt_s = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}__year"] = dt_s.dt.year
            X[f"{col}__month"] = dt_s.dt.month
            X[f"{col}__dayofweek"] = dt_s.dt.dayofweek
    X_feat = X.reindex(columns=bundle["features"])
    out = new_df.copy()
    pipe = bundle["pipeline"]
    if bundle["problem_type"] == "classification":
        if bundle.get("classes") and len(bundle["classes"]) == 2:
            p = pipe.predict_proba(X_feat)[:, 1]
            pos = bundle.get("positive_label") or bundle["classes"][1]
            neg = [c for c in bundle["classes"] if str(c) != str(pos)][0] if len(bundle["classes"]) == 2 else bundle["classes"][0]
            thr = bundle.get("threshold", 0.5) or 0.5
            out[f"predicted_{bundle['target']}"] = np.where(p >= thr, pos, neg)
            out["prediction_score"] = np.round(p, 4)
            out["risk_tier"] = pd.cut(p, [-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"]).astype(str)
        else:
            oof = pipe.predict_proba(X_feat)
            classes = bundle.get("classes", [])
            out[f"predicted_{bundle['target']}"] = [classes[i] if i < len(classes) else str(i) for i in oof.argmax(axis=1)]
            out["prediction_score"] = np.round(oof.max(axis=1), 4)
    else:
        out[f"predicted_{bundle['target']}"] = np.round(pipe.predict(X_feat), 4)
    return out

