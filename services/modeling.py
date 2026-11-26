"""Budget-aware supervised modeling with reproducible preprocessing and artifacts."""
from __future__ import annotations

import json
import os
import time
import gc
import warnings
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

try:
    import sys
    if sys.version_info >= (3, 14) and sys.platform == "win32":
        HAS_XGB = False
    else:
        from xgboost import XGBClassifier, XGBRegressor
        HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ENABLED_MODELS = [
    "Logistic Regression",
    "Ridge",
    "ElasticNet",
    "Lasso",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "HistGradientBoosting",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric, numeric_cols))
    if categorical_cols:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _schema_for(X: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> dict:
    columns = {}
    for col in numeric_cols:
        vals = pd.to_numeric(X[col], errors="coerce").dropna()
        columns[col] = {
            "type": str(X[col].dtype),
            "kind": "numeric",
            "min": float(vals.min()) if len(vals) else None,
            "max": float(vals.max()) if len(vals) else None,
        }
    for col in categorical_cols:
        categories = [str(v) for v in X[col].dropna().unique().tolist()]
        columns[col] = {
            "type": str(X[col].dtype),
            "kind": "categorical",
            "allowed_categories": sorted(categories),
        }
    return {
        "column_names": list(X.columns),
        "columns": columns,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def _detect_leakage(X: pd.DataFrame, y: pd.Series, is_classification: bool) -> list[str]:
    issues = []
    y_values = np.asarray(y)
    for col in X.columns:
        feature = X[col]
        valid = feature.notna()
        if int(valid.sum()) < 10:
            continue
        try:
            left = feature.loc[valid].astype(str).to_numpy()
            right = pd.Series(y, index=X.index).loc[valid].astype(str).to_numpy()
            if np.mean(left == right) > 0.98:
                issues.append(f"Feature '{col}' is a near-duplicate of the target (>0.98 exact agreement). Possible leakage.")
                continue
        except Exception:
            pass
        if pd.api.types.is_numeric_dtype(feature):
            pair = pd.DataFrame({"x": pd.to_numeric(feature, errors="coerce"), "y": y}, index=X.index).dropna()
            if len(pair) >= 10 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
                corr = abs(float(pair["x"].corr(pair["y"])))
                if np.isfinite(corr) and corr > 0.98:
                    issues.append(f"Feature '{col}' has absolute target correlation {corr:.4f} (>0.98). Possible leakage.")
        if is_classification and len(np.unique(y_values)) == 2:
            try:
                if pd.api.types.is_numeric_dtype(feature):
                    encoded = pd.to_numeric(feature, errors="coerce").fillna(feature.median())
                    auc = float(roc_auc_score(y_values, encoded))
                    auc = max(auc, 1.0 - auc)
                else:
                    labels = feature.astype("string").fillna("__MISSING__")
                    auc = max(
                        max(float(roc_auc_score(y_values, (labels == category).astype(int))),
                            1.0 - float(roc_auc_score(y_values, (labels == category).astype(int))))
                        for category in labels.unique()
                    )
                if auc > 0.98:
                    issues.append(f"Feature '{col}' has single-feature AUC {auc:.4f} (>0.98). Possible leakage.")
            except Exception:
                pass
    return list(dict.fromkeys(issues))


def _safe_score(estimator, X, y, cv, scoring) -> tuple[list[float], str | None]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        finite = [float(v) for v in values if np.isfinite(v)]
        if finite:
            return finite, None
        return [], "Cross-validation produced no finite scores."
    except Exception as exc:
        return [], str(exc)


def _macro_average_precision(estimator, X, y) -> float:
    """Macro one-vs-rest average precision for imbalanced multiclass targets."""
    probabilities = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    truth = label_binarize(y, classes=classes)
    if truth.shape[1] != probabilities.shape[1]:
        raise ValueError("Class probability columns do not match validation labels.")
    return float(average_precision_score(truth, probabilities, average="macro"))


def _minority_average_precision(estimator, X, y, positive_label) -> float:
    probabilities = estimator.predict_proba(X)
    class_index = list(estimator.classes_).index(positive_label)
    return float(average_precision_score(np.asarray(y) == positive_label, probabilities[:, class_index]))


def _get_process_mem_mb() -> str:
    try:
        import psutil
        mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        return f" ({mem_mb:.1f} MB)"
    except Exception:
        return ""


def _log_model(name: str, duration: float) -> None:
    print(f"[analyze] {name}: {duration:.2f}s{_get_process_mem_mb()}", flush=True)


def auto_model(df: pd.DataFrame, target: str, analysis_type: str = "auto",
               save_dir: str | None = "models", session_id: str = "default") -> dict:
    """Compare and persist models while retaining the legacy response contract."""
    started = time.monotonic()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if not target or target not in df.columns:
        return {"error": f"Target column '{target}' not present in dataframe."}
    data = df.dropna(subset=[target]).copy()
    if len(data) < 15:
        return {"error": f"Insufficient sample size ({len(data)} rows) for supervised modeling."}
    if data[target].nunique() < 2:
        return {"error": f"Target column '{target}' has only 1 unique value. Modeling requires at least 2 distinct classes or variance."}

    is_classification = data[target].nunique() <= 10
    problem_type = "classification" if is_classification else "regression"
    if is_classification:
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(data[target].astype(str)), index=data.index)
        classes = [str(v) for v in label_encoder.classes_]
    else:
        y = pd.to_numeric(data[target], errors="coerce")
        valid = y.notna()
        data, y = data.loc[valid], y.loc[valid]
        classes = None

    features = [c for c in data.columns if c != target]
    features = [c for c in features if not (
        any(c.lower().endswith(suffix) or c.lower() == suffix for suffix in ("id", "uuid", "guid", "key"))
        and data[c].nunique() / max(len(data), 1) > 0.6
    )]
    if not features:
        return {"error": "No valid predictor features available for training."}

    X = data[features].copy()
    leakage_warnings = _detect_leakage(X, y, is_classification)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    schema = _schema_for(X, numeric_cols, categorical_cols)
    subsampled = False

    if is_classification:
        minority = float(y.value_counts(normalize=True).min())
        use_pr = minority < 0.20
        metric = "PR-AUC" if use_pr else "ROC-AUC"
        metric_reason = (f"Minority class is {minority:.1%}, below 20%; average precision emphasizes positive-class retrieval."
                        if use_pr else f"Minority class is {minority:.1%}; ROC-AUC is used because no binary class is below 20%." )
        scoring = (partial(_minority_average_precision, positive_label=y.value_counts().idxmin())
               if len(classes) == 2 else _macro_average_precision) if use_pr else (
            "roc_auc" if len(classes) == 2 else "roc_auc_ovr_weighted"
        )
        if use_pr and len(classes) > 2:
            metric_reason = f"Smallest class is {minority:.1%}, below 20%; macro one-vs-rest average precision is used for multiclass imbalance."
        min_class = int(y.value_counts().min())
        folds = min(5, min_class)
        if folds < 2:
            return {"error": "At least two examples per class are required for stratified cross-validation."}
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        baseline_estimator = DummyClassifier(strategy="prior")
    else:
        metric, scoring = "R2", "r2"
        metric_reason = "R2 measures explained variance for a continuous target; MAE is reported as a complementary error scale."
        folds = min(5, max(2, len(y) // 3))
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        baseline_estimator = DummyRegressor(strategy="mean")

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    cv_results = {}
    candidates = []
    notes = []

    if is_classification:
        if "Logistic Regression" in ENABLED_MODELS:
            try:
                candidates.append(("Logistic Regression", LogisticRegression(max_iter=800, class_weight="balanced", random_state=42, n_jobs=1)))
            except Exception as exc:
                notes.append(f"Logistic Regression failed: Init failed: {exc}")
                cv_results["Logistic Regression"] = {"error": f"Init failed: {exc}"}
        if "Decision Tree" in ENABLED_MODELS:
            try:
                candidates.append(("Decision Tree", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)))
            except Exception as exc:
                notes.append(f"Decision Tree failed: Init failed: {exc}")
                cv_results["Decision Tree"] = {"error": f"Init failed: {exc}"}
        if "HistGradientBoosting" in ENABLED_MODELS:
            try:
                candidates.append(("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=60, class_weight="balanced", random_state=42)))
            except Exception as exc:
                notes.append(f"HistGradientBoosting failed: Init failed: {exc}")
                cv_results["HistGradientBoosting"] = {"error": f"Init failed: {exc}"}
        if "Random Forest" in ENABLED_MODELS:
            try:
                candidates.append(("Random Forest", RandomForestClassifier(n_estimators=50, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"Random Forest failed: Init failed: {exc}")
                cv_results["Random Forest"] = {"error": f"Init failed: {exc}"}
        if "Extra Trees" in ENABLED_MODELS:
            try:
                candidates.append(("Extra Trees", ExtraTreesClassifier(n_estimators=50, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"Extra Trees failed: Init failed: {exc}")
                cv_results["Extra Trees"] = {"error": f"Init failed: {exc}"}
        if "LightGBM" in ENABLED_MODELS and HAS_LGBM:
            try:
                candidates.append(("LightGBM", LGBMClassifier(n_estimators=60, class_weight="balanced", verbosity=-1, n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"LightGBM failed: Init failed: {exc}")
                cv_results["LightGBM"] = {"error": f"Init failed: {exc}"}
        if "XGBoost" in ENABLED_MODELS and HAS_XGB:
            try:
                weight = float((y == 0).sum() / max((y == 1).sum(), 1)) if len(classes) == 2 else 1.0
                candidates.append(("XGBoost", XGBClassifier(n_estimators=40, max_depth=4, learning_rate=0.08, scale_pos_weight=weight, eval_metric="logloss", verbosity=0, n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"XGBoost failed: Init failed: {exc}")
                cv_results["XGBoost"] = {"error": f"Init failed: {exc}"}
        if "CatBoost" in ENABLED_MODELS:
            if not HAS_CATBOOST:
                notes.append("CatBoost is not installed or failed to import.")
                cv_results["CatBoost"] = {"error": "CatBoost not installed or failed to import"}
            else:
                try:
                    try:
                        cb_clf = CatBoostClassifier(iterations=60, verbose=0, random_state=42, auto_class_weights="Balanced", thread_count=1)
                    except Exception:
                        cb_clf = CatBoostClassifier(iterations=60, verbose=0, random_state=42, thread_count=1)
                    candidates.append(("CatBoost", cb_clf))
                except Exception as exc:
                    notes.append(f"CatBoost failed: Init failed: {exc}")
                    cv_results["CatBoost"] = {"error": f"Init failed: {exc}"}
    else:
        if "Ridge" in ENABLED_MODELS:
            try:
                candidates.append(("Ridge", Ridge()))
            except Exception as exc:
                notes.append(f"Ridge failed: Init failed: {exc}")
                cv_results["Ridge"] = {"error": f"Init failed: {exc}"}
        if "ElasticNet" in ENABLED_MODELS:
            try:
                candidates.append(("ElasticNet", ElasticNet(random_state=42)))
            except Exception as exc:
                notes.append(f"ElasticNet failed: Init failed: {exc}")
                cv_results["ElasticNet"] = {"error": f"Init failed: {exc}"}
        if "Lasso" in ENABLED_MODELS:
            try:
                candidates.append(("Lasso", Lasso(random_state=42)))
            except Exception as exc:
                notes.append(f"Lasso failed: Init failed: {exc}")
                cv_results["Lasso"] = {"error": f"Init failed: {exc}"}
        if "Decision Tree" in ENABLED_MODELS:
            try:
                candidates.append(("Decision Tree", DecisionTreeRegressor(max_depth=4, random_state=42)))
            except Exception as exc:
                notes.append(f"Decision Tree failed: Init failed: {exc}")
                cv_results["Decision Tree"] = {"error": f"Init failed: {exc}"}
        if "HistGradientBoosting" in ENABLED_MODELS:
            try:
                candidates.append(("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=60, random_state=42)))
            except Exception as exc:
                notes.append(f"HistGradientBoosting failed: Init failed: {exc}")
                cv_results["HistGradientBoosting"] = {"error": f"Init failed: {exc}"}
        if "Random Forest" in ENABLED_MODELS:
            try:
                candidates.append(("Random Forest", RandomForestRegressor(n_estimators=50, min_samples_leaf=2, n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"Random Forest failed: Init failed: {exc}")
                cv_results["Random Forest"] = {"error": f"Init failed: {exc}"}
        if "Extra Trees" in ENABLED_MODELS:
            try:
                candidates.append(("Extra Trees", ExtraTreesRegressor(n_estimators=50, min_samples_leaf=2, n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"Extra Trees failed: Init failed: {exc}")
                cv_results["Extra Trees"] = {"error": f"Init failed: {exc}"}
        if "LightGBM" in ENABLED_MODELS and HAS_LGBM:
            try:
                candidates.append(("LightGBM", LGBMRegressor(n_estimators=60, verbosity=-1, n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"LightGBM failed: Init failed: {exc}")
                cv_results["LightGBM"] = {"error": f"Init failed: {exc}"}
        if "XGBoost" in ENABLED_MODELS and HAS_XGB:
            try:
                candidates.append(("XGBoost", XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.08, objective="reg:squarederror", verbosity=0, n_jobs=1, random_state=42)))
            except Exception as exc:
                notes.append(f"XGBoost failed: Init failed: {exc}")
                cv_results["XGBoost"] = {"error": f"Init failed: {exc}"}
        if "CatBoost" in ENABLED_MODELS:
            if not HAS_CATBOOST:
                notes.append("CatBoost is not installed or failed to import.")
                cv_results["CatBoost"] = {"error": "CatBoost not installed or failed to import"}
            else:
                try:
                    cb_reg = CatBoostRegressor(iterations=60, verbose=0, random_state=42, thread_count=1)
                    candidates.append(("CatBoost", cb_reg))
                except Exception as exc:
                    notes.append(f"CatBoost failed: Init failed: {exc}")
                    cv_results["CatBoost"] = {"error": f"Init failed: {exc}"}

    baseline_pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", baseline_estimator)])
    baseline_scores, baseline_error = _safe_score(baseline_pipe, X, y, cv, scoring)
    baseline_mean = float(np.mean(baseline_scores)) if baseline_scores else (0.5 if is_classification and metric == "ROC-AUC" else (minority if is_classification else 0.0))
    cv_results["Baseline"] = {"mean": round(baseline_mean, 4), "std": round(float(np.std(baseline_scores)), 4) if baseline_scores else 0.0}
    if baseline_error:
        cv_results["Baseline"]["error"] = baseline_error

    fitted_candidates = {}
    best_name, best_pipe, best_score = None, None, -float("inf")

    def evaluate(name, estimator):
        nonlocal best_name, best_pipe, best_score
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(estimator))])
        scores, error = _safe_score(pipe, X, y, cv, scoring)
        if error:
            cv_results[name] = {"error": error}
            notes.append(f"{name} evaluation failed: {error}")
            return None
        mean = float(np.mean(scores))
        cv_results[name] = {"mean": round(mean, 4), "std": round(float(np.std(scores)), 4)}
        fitted_candidates[name] = (pipe, mean)
        if mean > best_score:
            best_name, best_pipe, best_score = name, pipe, mean
        return mean

    for name, estimator in candidates:
        t_model = time.perf_counter()
        evaluate(name, estimator)
        _log_model(name, time.perf_counter() - t_model)
        gc.collect()

    if best_pipe is None:
        best_name, best_pipe, best_score = "Baseline", baseline_pipe, baseline_mean

    if not np.isfinite(best_score):
        best_score = baseline_mean
        best_name, best_pipe = "Baseline", baseline_pipe

    ensemble_kept = False
    ensemble_details = "Ensemble not evaluated."
    valid_singles = [
        (n, m, p) for n, (p, m) in fitted_candidates.items()
        if n != "Baseline" and np.isfinite(m)
    ]
    valid_singles.sort(key=lambda item: item[1], reverse=True)

    if len(valid_singles) >= 3:
        top_3 = valid_singles[:3]
        top_3_names = [item[0] for item in top_3]
        t_ens = time.perf_counter()
        try:
            member_estimators = [(n, clone(p.named_steps["model"])) for n, _, p in top_3]
            if is_classification:
                ensemble_model = VotingClassifier(estimators=member_estimators, voting="soft", n_jobs=1)
            else:
                ensemble_model = VotingRegressor(estimators=member_estimators, n_jobs=1)

            ensemble_pipe = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", ensemble_model)
            ])

            ens_scores, ens_error = _safe_score(ensemble_pipe, X, y, cv, scoring)
            if ens_error:
                cv_results["Ensemble"] = {"error": ens_error}
                ensemble_details = f"Ensemble evaluation failed: {ens_error}"
                notes.append(ensemble_details)
            elif ens_scores:
                ens_mean = float(np.mean(ens_scores))
                ens_std = float(np.std(ens_scores))
                cv_results["Ensemble"] = {
                    "mean": round(ens_mean, 4),
                    "std": round(ens_std, 4)
                }

                best_single_name = valid_singles[0][0]
                best_single_score = valid_singles[0][1]

                if ens_mean > best_single_score:
                    best_name = "Ensemble"
                    best_pipe = ensemble_pipe
                    best_score = ens_mean
                    ensemble_kept = True
                    ensemble_details = (
                        f"{'Soft-voting' if is_classification else 'Averaging'} ensemble of top 3 models "
                        f"({', '.join(top_3_names)}) kept: cross-validation {metric} ({ens_mean:.4f}) "
                        f"beat the best single model ({best_single_name}: {best_single_score:.4f})."
                    )
                else:
                    ensemble_kept = False
                    ensemble_details = (
                        f"Ensemble of top 3 models ({', '.join(top_3_names)}) evaluated "
                        f"(CV {metric}: {ens_mean:.4f}) but not kept: "
                        f"did not beat best single model ({best_single_name}: {best_single_score:.4f})."
                    )
                notes.append(ensemble_details)
        except Exception as exc:
            cv_results["Ensemble"] = {"error": str(exc)}
            ensemble_details = f"Ensemble creation/evaluation encountered error: {exc}"
            notes.append(ensemble_details)
        finally:
            _log_model("Ensemble", time.perf_counter() - t_ens)
            gc.collect()
    else:
        ensemble_details = f"Ensemble omitted: fewer than 3 single models were successfully evaluated ({len(valid_singles)})."
        notes.append(ensemble_details)

    brier = None
    calibrated_pipe = best_pipe
    try:
        best_pipe.fit(X, y)
        if is_classification:
            min_count = int(y.value_counts().min())
            cal_folds = min(3, min_count)
            if cal_folds >= 2:
                try:
                    try:
                        calibrator = CalibratedClassifierCV(estimator=clone(best_pipe), cv=cal_folds, n_jobs=1)
                    except TypeError:
                        calibrator = CalibratedClassifierCV(base_estimator=clone(best_pipe), cv=cal_folds)
                    calibrator.fit(X, y)
                    calibrated_pipe = calibrator
                    probs = calibrated_pipe.predict_proba(X)
                    if len(classes) == 2:
                        brier = round(float(brier_score_loss(y, probs[:, 1])), 4)
                    else:
                        brier = round(float(np.mean([brier_score_loss((y == i).astype(int), probs[:, i]) for i in range(len(classes))])), 4)
                except Exception as exc:
                    notes.append(f"Probability calibration failed; using the uncalibrated fitted pipeline ({exc}).")
    except Exception as exc:
        notes.append(f"Winner refit failed ({exc}); fitting the baseline instead.")
        best_name = "Baseline"
        best_pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(baseline_estimator))]).fit(X, y)
        calibrated_pipe = best_pipe
        best_score = baseline_mean

    stratify = y if is_classification and int(y.value_counts().min()) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        test_preds = best_pipe.predict(X_test)
    except Exception:
        test_preds = np.repeat(y_train.mode().iloc[0] if is_classification else y_train.mean(), len(y_test))
    cm, report, threshold, min_cost = [], {}, 0.5, 0
    if is_classification:
        cm = confusion_matrix(y_test, test_preds).tolist()
        report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
        try:
            probs = calibrated_pipe.predict_proba(X_test)
            positive_probs = probs[:, 1] if len(classes) == 2 else probs.max(axis=1)
            if len(classes) == 2:
                costs = []
                for candidate_threshold in np.arange(0.1, 0.95, 0.05):
                    decisions = (positive_probs >= candidate_threshold).astype(int)
                    fn = int(np.sum((y_test == 1) & (decisions == 0)))
                    fp = int(np.sum((y_test == 0) & (decisions == 1)))
                    costs.append((fn * 500 + fp * 50, candidate_threshold))
                min_cost, threshold = min(costs, key=lambda item: item[0])
                threshold = round(float(threshold), 2)
        except Exception:
            pass
    mae = None
    if not is_classification:
        try:
            mae = round(float(mean_absolute_error(y_test, test_preds)), 4)
        except Exception:
            pass

    baseline_spread = cv_results["Baseline"]["std"]
    margin = best_score - baseline_mean
    best_spread = cv_results.get(best_name, {}).get("std", 0.0)
    noise_floor = max(0.10, 2.0 * max(float(best_spread or 0.0), float(baseline_spread or 0.0)))
    verdict = "strong" if margin >= max(0.15, 3.0 * float(best_spread or 0.0)) else (
        "moderate" if margin >= noise_floor else "weak: barely beats baseline"
    )
    if best_name == "Baseline" or margin < 0:
        verdict = "weak: barely beats baseline"

    if best_name == "Ensemble":
        why_won = (
            f"{'Soft-voting' if is_classification else 'Averaging'} ensemble of top 3 models "
            f"({', '.join(top_3_names)}) achieved highest cross-validated {metric} ({best_score:.4f}); "
            f"improvement over baseline is {margin:+.4f}."
        )
    else:
        why_won = (f"Highest cross-validated {metric} ({best_score:.4f}); improvement over baseline is {margin:+.4f}."
                   if best_name != "Baseline" else "No non-baseline candidate produced a better valid cross-validation score.")

    decision_rules = None
    if "Decision Tree" in fitted_candidates:
        try:
            dt_pipe_proto, _ = fitted_candidates["Decision Tree"]
            dt_pipe_fit = clone(dt_pipe_proto)
            dt_pipe_fit.fit(X, y)
            prep_step = dt_pipe_fit.named_steps["preprocessor"]
            feature_names = [f.split("__", 1)[-1] for f in prep_step.get_feature_names_out()]
            dt_model_step = dt_pipe_fit.named_steps["model"]
            raw_rules = export_text(dt_model_step, feature_names=list(feature_names), max_depth=4)
            rule_lines = [line for line in raw_rules.splitlines() if line.strip()]
            decision_rules = "\n".join(rule_lines[:15])
        except Exception as exc:
            decision_rules = f"Could not export decision rules: {exc}"

    importance = {}
    try:
        model = best_pipe.named_steps.get("model")
        if model is not None:
            prep = best_pipe.named_steps["preprocessor"]
            names = [name.split("__", 1)[-1] for name in prep.get_feature_names_out()]
            values = getattr(model, "feature_importances_", None)
            if values is None and hasattr(model, "coef_"):
                coef = np.asarray(model.coef_)
                values = np.abs(coef[0] if coef.ndim > 1 else coef)
            if values is None and hasattr(model, "estimators_"):
                member_importances = []
                for est in model.estimators_:
                    sub_vals = getattr(est, "feature_importances_", None)
                    if sub_vals is None and hasattr(est, "coef_"):
                        c = np.asarray(est.coef_)
                        sub_vals = np.abs(c[0] if c.ndim > 1 else c)
                    if sub_vals is not None and len(sub_vals) == len(names):
                        s = float(np.sum(sub_vals))
                        if s > 0:
                            sub_vals = np.asarray(sub_vals, dtype=float) / s
                        member_importances.append(sub_vals)
                if member_importances:
                    values = np.mean(member_importances, axis=0)
            if values is not None:
                importance = {k: round(float(v), 4) for k, v in sorted(zip(names, values), key=lambda x: x[1], reverse=True)[:10]}
    except Exception:
        pass

    for warning_text in leakage_warnings:
        notes.insert(0, warning_text)
    model_card = {
        "dataset_size": int(len(X)), "features_used": list(X.columns),
        "models_tried": list(cv_results.keys()), "cv_scores": cv_results,
        "baseline": {"mean": round(baseline_mean, 4), "std": baseline_spread},
        "baseline_score": round(baseline_mean, 4), "metric": metric,
        "metric_reason": metric_reason, "winner": best_name,
        "winning_score": round(float(best_score), 4), "why_it_won": why_won,
        "decision_rules": decision_rules,
        "ensemble_kept": ensemble_kept,
        "ensemble_details": ensemble_details,
        "brier_score": brier, "mae": mae,
        "time_used_seconds": round(time.monotonic() - started, 2),
        "time_budget_seconds": None, "subsampled": False,
        "warnings": notes, "reliability_verdict": verdict,
    }
    bundle = {
        "pipeline": calibrated_pipe, "raw_pipeline": best_pipe,
        "features": list(X.columns), "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols, "target": target,
        "problem_type": problem_type, "threshold": threshold,
        "classes": classes, "analysis_type": analysis_type,
        "schema": schema, "model_card": model_card,
    }

    import io
    bundle_buf = io.BytesIO()
    joblib.dump(bundle, bundle_buf)
    bundle_bytes = bundle_buf.getvalue()

    model_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{session_id}_model.pkl")
        schema_path = os.path.join(save_dir, f"{session_id}_schema.json")
        try:
            with open(model_path, "wb") as f:
                f.write(bundle_bytes)
            with open(schema_path, "w", encoding="utf-8") as file:
                json.dump(schema, file, indent=2, ensure_ascii=False)
        except Exception as exc:
            notes.append(f"Could not save model artifact or schema: {exc}")

    lr_score = cv_results.get("Logistic Regression", {}).get("mean", cv_results.get("Ridge", {}).get("mean", 0.0))
    rf_score = cv_results.get("Random Forest", {}).get("mean", 0.0)
    xgb_score = cv_results.get("XGBoost", {}).get("mean", 0.0)
    dt_score = cv_results.get("Decision Tree", {}).get("mean", 0.0)
    et_score = cv_results.get("Extra Trees", {}).get("mean", 0.0)
    cb_score = cv_results.get("CatBoost", {}).get("mean", 0.0)
    ens_score = cv_results.get("Ensemble", {}).get("mean", 0.0)
    accuracy = round(float(accuracy_score(y_test, test_preds)) * 100, 2) if is_classification else round(float(best_score * 100), 2)
    return {
        "problem_type": problem_type, "best_model": best_name,
        "accuracy": accuracy, "accuracy_score": accuracy,
        "auc": round(float(best_score), 4), "auc_score": round(float(best_score), 4),
        "r2": round(float(best_score), 4) if not is_classification else None,
        "r2_score": round(float(best_score), 4) if not is_classification else None,
        "lr_score": round(float(lr_score), 4) if isinstance(lr_score, (int, float)) else 0.0,
        "rf_score": round(float(rf_score), 4) if isinstance(rf_score, (int, float)) else 0.0,
        "xgb_score": round(float(xgb_score), 4) if isinstance(xgb_score, (int, float)) else 0.0,
        "dt_score": round(float(dt_score), 4) if isinstance(dt_score, (int, float)) else 0.0,
        "et_score": round(float(et_score), 4) if isinstance(et_score, (int, float)) else 0.0,
        "cb_score": round(float(cb_score), 4) if isinstance(cb_score, (int, float)) else 0.0,
        "ensemble_score": round(float(ens_score), 4) if isinstance(ens_score, (int, float)) else 0.0,
        "decision_rules": decision_rules,
        "ensemble_kept": ensemble_kept,
        "cv_mean": round(float(best_score), 4), "cv_std": cv_results.get(best_name, {}).get("std", 0.0),
        "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
        "n_features": len(X.columns), "feature_importance": importance,
        "confusion_matrix": cm, "optimal_threshold": threshold,
        "min_cost": min_cost, "classification_report": report,
        "features_used": list(X.columns), "classes": classes,
        "model_path": model_path, "model_card": model_card,
        "bundle": bundle, "bundle_bytes": bundle_bytes,
        "brier_score": brier, "leakage_warnings": leakage_warnings,
        "schema": schema, "mae": mae, "metric": metric,
    }
