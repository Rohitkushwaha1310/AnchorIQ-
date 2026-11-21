"""Exploratory Data Analysis (EDA) automated charting service.
Renders statistical figures to in-memory PNG buffers or disk.
"""
import io
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "anchoriq_mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_or_buffer(fig, title: str, save_dir: str | None, filename: str):
    """Save matplotlib figure to in-memory bytes buffer, and optionally to disk if save_dir is specified."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    data = buf.getvalue()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename).replace("\\", "/")
        with open(path, "wb") as f:
            f.write(data)
        return path
    return {"title": title, "bytes": data}


def auto_eda(
    df: pd.DataFrame,
    target: str = None,
    save_dir: str = None,
    analysis_type: str = "auto",
    rf_model: dict = None,
) -> list:
    """Auto-generate comprehensive EDA charts for any business dataset.
    Returns list of saved file paths if save_dir is given, otherwise list of dicts with title and PNG bytes.
    """
    charts = []
    sns.set_style("whitegrid")

    nums_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if target in nums_cols:
        nums_cols.remove(target)
    if target in cat_cols:
        cat_cols.remove(target)

    # Chart 1: Feature Distributions
    if nums_cols:
        cols = nums_cols[:6]
        n = len(cols)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, col in enumerate(cols):
            axes[i].hist(df[col].dropna(), bins=30, color="#1F6B58", edgecolor="white", alpha=0.85)
            axes[i].set_title(col)
        for j in range(n, 6):
            axes[j].set_visible(False)
        plt.tight_layout()
        charts.append(_save_or_buffer(fig, "Feature Distributions", save_dir, "01_distribution.png"))

    # Chart 2: Correlation Heatmap
    if len(nums_cols) >= 2:
        corr_cols = nums_cols[:10]
        if target and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]):
                corr_cols = corr_cols + [target]
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        charts.append(_save_or_buffer(fig, "Correlation Heatmap", save_dir, "02_correlation.png"))

    # Chart 3: Target Distribution
    if target and target in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        if df[target].nunique() <= 10:
            counts = df[target].value_counts()
            colors = ["#1F6B58", "#B4532A", "#B8892B", "#6F7C5E"]
            ax.bar(counts.index.astype(str), counts.values, color=colors[: len(counts)])
            for i, v in enumerate(counts.values):
                ax.text(i, v + (max(counts.values) * 0.02), str(v), ha="center", fontweight="bold")
        else:
            df[target].hist(bins=30, color="#1F6B58", edgecolor="white", ax=ax)
        ax.set_title(f"Target: {target} Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        plt.tight_layout()
        charts.append(_save_or_buffer(fig, "Target Distribution", save_dir, "03_target.png"))

    # Chart 4: Categorical Impact vs Target
    if target and cat_cols:
        n = min(len(cat_cols), 4)
        cols = cat_cols[:n]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Features vs {target}", fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, col in enumerate(cols):
            try:
                mean_rate = df.groupby(col)[target].mean() * 100
                axes[i].bar(mean_rate.index.astype(str), mean_rate.values, color="#1F6B58")
                axes[i].set_title(f"{col} vs {target}")
                axes[i].tick_params(axis="x", rotation=15)
                axes[i].set_ylabel(f"{target} rate %")
            except Exception:
                pass
        for j in range(n, 4):
            axes[j].set_visible(False)
        plt.tight_layout()
        charts.append(_save_or_buffer(fig, "Categorical Impact", save_dir, "04_categorical_vs_target.png"))

    # Chart 5: Numerical Boxplots
    if nums_cols and target:
        cols = nums_cols[:4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Feature Distributions by Target", fontsize=14, fontweight="bold")
        axes = axes.flatten()
        for i, col in enumerate(cols):
            try:
                df.boxplot(column=col, by=target, ax=axes[i])
                axes[i].set_title(col)
            except Exception:
                pass
        plt.tight_layout()
        charts.append(_save_or_buffer(fig, "Numerical Boxplots", save_dir, "05_boxplots.png"))

    return charts
