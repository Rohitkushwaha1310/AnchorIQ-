"""Statistical A/B testing service supporting binary and continuous metrics across 2 or more groups."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def _is_binary_metric(series: pd.Series) -> bool:
    """Detect if a series represents a binary metric (booleans, 0/1, or binary labels)."""
    clean = series.dropna()
    if len(clean) == 0:
        return False
    if pd.api.types.is_bool_dtype(clean):
        return True
    unique_vals = list(clean.unique())
    if len(unique_vals) > 2:
        return False
    val_set = set(unique_vals)
    if val_set.issubset({0, 1, 0.0, 1.0, True, False}):
        return True
    str_vals = {str(v).strip().lower() for v in unique_vals}
    binary_keywords = {
        "0", "1", "true", "false", "yes", "no", "y", "n",
        "success", "failure", "churn", "retained", "converted", "bounced"
    }
    if str_vals.issubset(binary_keywords):
        return True
    if pd.api.types.is_numeric_dtype(clean):
        return False
    return True


def run_ab_test(
    df: Optional[pd.DataFrame] = None,
    group_column: Optional[str] = None,
    metric_column: Optional[str] = None,
    group_a_df: Optional[pd.DataFrame] = None,
    group_b_df: Optional[pd.DataFrame] = None,
    metric_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform statistical A/B test analysis.
    Supports:
      1. Single DataFrame with group_column and metric_column
      2. Legacy two DataFrames (group_a_df, group_b_df) with metric_col
    """
    if df is None:
        if group_a_df is not None and group_b_df is not None:
            col = metric_column or metric_col
            if not col:
                raise ValueError("Metric column must be provided.")
            g_a = group_a_df.copy()
            g_b = group_b_df.copy()
            g_a["_ab_group"] = "Control (A)"
            g_b["_ab_group"] = "Treatment (B)"
            df = pd.concat([g_a, g_b], ignore_index=True)
            group_column = "_ab_group"
            metric_column = col
        else:
            raise ValueError("Dataset or group DataFrames must be provided.")

    if not group_column or group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in dataset.")
    if not metric_column or metric_column not in df.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in dataset.")

    valid_df = df.dropna(subset=[group_column, metric_column]).copy()
    distinct_groups = list(valid_df[group_column].unique())
    if len(distinct_groups) < 2:
        raise ValueError(f"Group column '{group_column}' has only {len(distinct_groups)} group(s). At least 2 groups required.")

    group_sizes = {str(g): int((valid_df[group_column] == g).sum()) for g in distinct_groups}
    warnings_list: List[str] = []
    small_groups = [g for g, n in group_sizes.items() if n < 30]
    if small_groups:
        warnings_list.append(
            f"Small sample size (< 30) detected in group(s): {', '.join(small_groups)}. "
            "Statistical power may be limited, so interpret results with caution."
        )

    causation_note = "Observational data does not prove causation; unobserved confounders may influence these results."

    is_binary = _is_binary_metric(valid_df[metric_column])

    if is_binary:
        pos_values = {1, "1", True, "true", "yes", "y", "positive", "churn"}
        raw_vals = valid_df[metric_column].tolist()
        val_set = set(raw_vals)
        if any(v in pos_values for v in val_set):
            binary_series = valid_df[metric_column].apply(
                lambda v: 1 if v in pos_values or str(v).lower() in {"1", "true", "yes", "y"} else 0
            )
        else:
            sorted_classes = sorted(list(val_set), key=lambda x: str(x))
            pos_label = sorted_classes[-1]
            binary_series = (valid_df[metric_column] == pos_label).astype(int)

        valid_df["_binary_metric"] = binary_series

        if len(distinct_groups) == 2:
            g1, g2 = distinct_groups[0], distinct_groups[1]
            s1 = valid_df[valid_df[group_column] == g1]["_binary_metric"]
            s2 = valid_df[valid_df[group_column] == g2]["_binary_metric"]

            n1, n2 = len(s1), len(s2)
            x1, x2 = int(s1.sum()), int(s2.sum())
            p1 = float(x1 / n1) if n1 > 0 else 0.0
            p2 = float(x2 / n2) if n2 > 0 else 0.0

            table = [[x1, n1 - x1], [x2, n2 - x2]]
            try:
                chi2_stat, p_val, _, _ = stats.chi2_contingency(table, correction=False)
            except Exception:
                chi2_stat, p_val = 0.0, 1.0

            diff = p2 - p1
            lift_pct = (diff / p1 * 100.0) if p1 > 0 else 0.0
            se = np.sqrt((p1 * (1.0 - p1) / max(n1, 1)) + (p2 * (1.0 - p2) / max(n2, 1)))
            ci_low = float(diff - 1.96 * se)
            ci_high = float(diff + 1.96 * se)

            is_sig = bool(p_val < 0.05)
            if is_sig:
                direction = "higher" if diff > 0 else "lower"
                verdict = (
                    f"Statistically significant difference detected (p = {p_val:.4f}). "
                    f"'{g2}' success rate ({p2:.1%}) is {direction} than '{g1}' ({p1:.1%}) "
                    f"with an estimated lift of {lift_pct:+.2f}% (95% CI: [{ci_low:.3f}, {ci_high:.3f}])."
                )
            else:
                verdict = (
                    f"No statistically significant difference detected (p = {p_val:.4f}). "
                    f"Rates for '{g1}' ({p1:.1%}) and '{g2}' ({p2:.1%}) are within expected random variance."
                )

            return {
                "metric_type": "binary",
                "test_type": "Two-proportion Z-test / Chi-square",
                "p_value": round(float(p_val), 4),
                "significant": is_sig,
                "effect_size": round(float(lift_pct), 2),
                "effect_size_label": "Lift (%)",
                "lift_pct": round(float(lift_pct), 2),
                "cohens_d": None,
                "confidence_interval": [round(ci_low, 4), round(ci_high, 4)],
                "group_sizes": group_sizes,
                "group_stats": {
                    str(g1): {"n": n1, "successes": x1, "rate": round(p1, 4)},
                    str(g2): {"n": n2, "successes": x2, "rate": round(p2, 4)},
                },
                "plain_english_verdict": verdict,
                "warnings": warnings_list,
                "notes": [causation_note],
            }
        else:
            table = []
            group_stats = {}
            for g in distinct_groups:
                s = valid_df[valid_df[group_column] == g]["_binary_metric"]
                succ = int(s.sum())
                total = len(s)
                table.append([succ, total - succ])
                group_stats[str(g)] = {"n": total, "successes": succ, "rate": round(succ / max(total, 1), 4)}

            try:
                chi2_stat, p_val, dof, _ = stats.chi2_contingency(table)
                n_total = len(valid_df)
                cramers_v = np.sqrt(chi2_stat / (n_total * max(len(distinct_groups) - 1, 1))) if n_total > 0 else 0.0
            except Exception:
                p_val, cramers_v = 1.0, 0.0

            is_sig = bool(p_val < 0.05)
            rates = [gs["rate"] for gs in group_stats.values()]
            if is_sig:
                verdict = (
                    f"Statistically significant difference detected across {len(distinct_groups)} groups (Chi-square p = {p_val:.4f}, Cramér's V = {cramers_v:.3f}). "
                    f"Rates vary from {min(rates):.1%} to {max(rates):.1%}."
                )
            else:
                verdict = (
                    f"No statistically significant difference detected across groups (Chi-square p = {p_val:.4f}). "
                    "Variation across groups is consistent with chance."
                )

            return {
                "metric_type": "binary",
                "test_type": "Chi-square Test of Independence",
                "p_value": round(float(p_val), 4),
                "significant": is_sig,
                "effect_size": round(float(cramers_v), 4),
                "effect_size_label": "Cramér's V",
                "lift_pct": None,
                "cohens_d": None,
                "confidence_interval": None,
                "group_sizes": group_sizes,
                "group_stats": group_stats,
                "plain_english_verdict": verdict,
                "warnings": warnings_list,
                "notes": [causation_note],
            }

    else:
        num_series = pd.to_numeric(valid_df[metric_column], errors="coerce")
        valid_df["_num_metric"] = num_series
        valid_df = valid_df.dropna(subset=["_num_metric"])

        if len(distinct_groups) == 2:
            g1, g2 = distinct_groups[0], distinct_groups[1]
            s1 = valid_df[valid_df[group_column] == g1]["_num_metric"].to_numpy(dtype=float)
            s2 = valid_df[valid_df[group_column] == g2]["_num_metric"].to_numpy(dtype=float)

            n1, n2 = len(s1), len(s2)
            if n1 < 2 or n2 < 2:
                raise ValueError("Both groups must have at least 2 valid numeric data points.")

            m1, m2 = float(np.mean(s1)), float(np.mean(s2))
            v1, v2 = float(np.var(s1, ddof=1)), float(np.var(s2, ddof=1))
            med1, med2 = float(np.median(s1)), float(np.median(s2))

            t_stat, p_welch = stats.ttest_ind(s2, s1, equal_var=False)
            if np.isnan(p_welch):
                p_welch = 1.0

            try:
                u_stat, p_mw = stats.mannwhitneyu(s2, s1, alternative="two-sided")
            except Exception:
                p_mw = 1.0

            diff = m2 - m1
            denom = (n1 + n2 - 2)
            pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / denom) if denom > 0 else 1.0
            cohens_d = float(diff / pooled_sd) if pooled_sd > 0 else 0.0
            lift_pct = (diff / abs(m1) * 100.0) if m1 != 0 else 0.0

            se = np.sqrt((v1 / n1) + (v2 / n2))
            df_num = ((v1 / n1) + (v2 / n2)) ** 2
            df_denom = ((v1 / n1) ** 2 / max(n1 - 1, 1)) + ((v2 / n2) ** 2 / max(n2 - 1, 1))
            df_welch = df_num / df_denom if df_denom > 0 else 1.0
            t_crit = float(stats.t.ppf(0.975, df_welch)) if df_welch > 0 else 1.96
            ci_low = float(diff - t_crit * se)
            ci_high = float(diff + t_crit * se)

            is_sig = bool(p_welch < 0.05)
            if is_sig:
                dir_str = "higher" if diff > 0 else "lower"
                verdict = (
                    f"Statistically significant difference detected (Welch's t-test p = {p_welch:.4f}, Mann-Whitney p = {p_mw:.4f}). "
                    f"'{g2}' average ({m2:.2f}) is {dir_str} than '{g1}' ({m1:.2f}) "
                    f"with Cohen's d of {cohens_d:.2f} ({lift_pct:+.2f}% lift, 95% CI: [{ci_low:.2f}, {ci_high:.2f}])."
                )
            else:
                verdict = (
                    f"No statistically significant difference detected (Welch's t-test p = {p_welch:.4f}, Mann-Whitney p = {p_mw:.4f}). "
                    f"Difference between '{g1}' (mean: {m1:.2f}) and '{g2}' (mean: {m2:.2f}) is consistent with chance."
                )

            return {
                "metric_type": "continuous",
                "test_type": "Welch's t-test + Mann-Whitney",
                "p_value": round(float(p_welch), 4),
                "p_value_welch": round(float(p_welch), 4),
                "p_value_mann_whitney": round(float(p_mw), 4),
                "significant": is_sig,
                "effect_size": round(float(cohens_d), 4),
                "effect_size_label": "Cohen's d",
                "lift_pct": round(float(lift_pct), 2),
                "cohens_d": round(float(cohens_d), 4),
                "confidence_interval": [round(ci_low, 4), round(ci_high, 4)],
                "group_sizes": group_sizes,
                "group_stats": {
                    str(g1): {"n": n1, "mean": round(m1, 4), "median": round(med1, 4), "std": round(float(np.sqrt(v1)), 4)},
                    str(g2): {"n": n2, "mean": round(m2, 4), "median": round(med2, 4), "std": round(float(np.sqrt(v2)), 4)},
                },
                "plain_english_verdict": verdict,
                "warnings": warnings_list,
                "notes": [causation_note],
            }
        else:
            series_list = [valid_df[valid_df[group_column] == g]["_num_metric"].dropna().to_numpy(dtype=float) for g in distinct_groups]
            group_stats = {}
            for g, s in zip(distinct_groups, series_list):
                group_stats[str(g)] = {
                    "n": len(s),
                    "mean": round(float(np.mean(s)), 4) if len(s) else 0.0,
                    "median": round(float(np.median(s)), 4) if len(s) else 0.0,
                    "std": round(float(np.std(s, ddof=1)), 4) if len(s) > 1 else 0.0,
                }

            f_stat, p_anova = stats.f_oneway(*series_list)
            try:
                h_stat, p_kw = stats.kruskal(*series_list)
            except Exception:
                p_kw = 1.0

            all_vals = np.concatenate(series_list)
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(s) * (np.mean(s) - grand_mean) ** 2 for s in series_list)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

            is_sig = bool(p_anova < 0.05)
            means = [gs["mean"] for gs in group_stats.values()]
            if is_sig:
                verdict = (
                    f"Statistically significant difference detected across {len(distinct_groups)} groups "
                    f"(ANOVA p = {p_anova:.4f}, Kruskal-Wallis p = {p_kw:.4f}, Eta-squared = {eta_squared:.3f}). "
                    f"Group means range from {min(means):.2f} to {max(means):.2f}."
                )
            else:
                verdict = (
                    f"No statistically significant difference detected across groups (ANOVA p = {p_anova:.4f}, Kruskal-Wallis p = {p_kw:.4f}). "
                    "Group variations are consistent with random chance."
                )

            return {
                "metric_type": "continuous",
                "test_type": "ANOVA / Kruskal-Wallis",
                "p_value": round(float(p_anova), 4),
                "p_value_anova": round(float(p_anova), 4),
                "p_value_kruskal_wallis": round(float(p_kw), 4),
                "significant": is_sig,
                "effect_size": round(float(eta_squared), 4),
                "effect_size_label": "Eta-squared",
                "lift_pct": None,
                "cohens_d": None,
                "confidence_interval": None,
                "group_sizes": group_sizes,
                "group_stats": group_stats,
                "plain_english_verdict": verdict,
                "warnings": warnings_list,
                "notes": [causation_note],
            }
