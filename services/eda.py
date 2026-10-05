import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def auto_eda(df: pd.DataFrame,
             target: str = None,
             save_dir: str = 'charts',
             analysis_type: str = 'auto') -> list:
    """
    Auto generate EDA charts for ANY dataset.
    Returns list of saved chart paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    charts = []

    sns.set_style('whitegrid')

    nums_cols = df.select_dtypes(
        include = [np.number]).columns.tolist()
    cat_cols = df.select_dtypes(
        include=['object', 'category', 'string']
    ).columns.tolist()
    # only columns that make a readable chart (skips IDs / free text)
    cat_cols = [c for c in cat_cols if 2 <= df[c].nunique() <= 15]

    if target in nums_cols:
        nums_cols.remove(target)
    
    if target in cat_cols:
        cat_cols.remove(target)

    #chart 1 distribution
    if nums_cols:
        cols = nums_cols[:6]
        n= len(cols)
        fig,axes = plt.subplots(2,3, figsize = (15,8))
        fig.suptitle('Feature Distributions',
                     fontsize=14, fontweight='bold')    

        axes = axes.flatten()
        for i , col in enumerate(cols):
            axes[i].hist(df[col].dropna(), bins=30,
                         color='steelblue',
                         edgecolor='white', alpha=0.8)
            axes[i].set_title(col)

        for j in range(n,6):
            axes[j].set_visible(False)

        plt.tight_layout()
        path = f'{save_dir}/01_distribution.png'  
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        charts.append(path)
        print(f"[OK] Chart 1: Distributions saved")
    #chart 2 correlation heap
    if len(nums_cols)>=2:
        corr_cols = nums_cols[:10]
        if target and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]):
                corr_cols = corr_cols + [target]
        plt.figure(figsize=(10, 7))
        sns.heatmap(df[corr_cols].corr(),
                    annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = f'{save_dir}/02_correlation.png'
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        charts.append(path)
        print(f"[OK] Chart 2: Correlation saved")


    #chart 3 : targte distribbution    
    if target and target in df.columns:
        plt.figure(figsize=(8, 5))
        if df[target].nunique() <= 10:
            counts = df[target].value_counts()
            colors = ['#2ecc71','#e74c3c',
                      '#3498db','#f39c12']
            plt.bar(counts.index.astype(str),
                    counts.values,
                    color=colors[:len(counts)])
            for i, v in enumerate(counts.values):
                plt.text(i, v+10, str(v),
                         ha='center', fontweight='bold')
        else:
            df[target].hist(bins=30,
                            color='steelblue',
                            edgecolor='white')
        plt.title(f'Target: {target} Distribution',
                  fontsize=14, fontweight='bold')
        plt.xlabel(target)
        plt.ylabel('Count')
        plt.tight_layout()
        path = f'{save_dir}/03_target.png'
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        charts.append(path)
        print(f"[OK] Chart 3: Target distribution saved")


    # chart 4: categories vs target (numeric/binary target -> mean rate; text target -> stacked share)
    if target and cat_cols:
        n    = min(len(cat_cols), 4)
        cols = cat_cols[:n]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Features vs {target}',
                     fontsize=14, fontweight='bold')
        axes = axes.flatten()
        drawn = 0
        target_is_num = pd.api.types.is_numeric_dtype(df[target])
        for i, col in enumerate(cols):
            try:
                if target_is_num:
                    rate = df.groupby(col)[target].mean()
                    if df[target].nunique() <= 2:
                        rate = rate * 100
                    axes[i].bar(rate.index.astype(str), rate.values, color='steelblue')
                    axes[i].set_ylabel(f'{target} rate %' if df[target].nunique() <= 2 else f'avg {target}')
                else:
                    share = pd.crosstab(df[col], df[target], normalize='index') * 100
                    share.plot(kind='bar', stacked=True, ax=axes[i], legend=(i == 0))
                    axes[i].set_ylabel(f'% by {target}')
                axes[i].set_title(f'{col} vs {target}')
                axes[i].tick_params(axis='x', rotation=15)
                drawn += 1
            except Exception as e:
                print(f"[WARN] chart 4 panel '{col}' skipped: {e}")
                axes[i].set_visible(False)
        for j in range(n, 4):
            axes[j].set_visible(False)
        if drawn:
            plt.tight_layout()
            path = f'{save_dir}/04_categorical_vs_target.png'
            plt.savefig(path, dpi=120, bbox_inches='tight')
            charts.append(path)
            print(f"[OK] Chart 4: Categorical vs target saved")
        plt.close()
    #chart 5 boxplots
    if nums_cols and target and 2 <= df[target].nunique() <= 10:
        cols = nums_cols[:4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Feature Distributions by Target',
                     fontsize=14, fontweight='bold')
        axes = axes.flatten()
        for i, col in enumerate(cols):
            try:
                df.boxplot(column=col,
                           by=target, ax=axes[i])
                axes[i].set_title(col)
            except Exception as e:
                print(f"[WARN] boxplot '{col}' skipped: {e}")
        plt.tight_layout()
        path = f'{save_dir}/05_boxplots.png'
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        charts.append(path)
        print(f"[OK] Chart 5: Boxplots saved")

    print(f"\n[OK] Total charts generated: {len(charts)}")
    return charts


if __name__ == "__main__":
    df = pd.read_csv("telco_churn.csv")
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(
        df['TotalCharges'].median(), inplace=True)
    df['Churn_Binary'] = (df['Churn']=='Yes').astype(int)

    charts = auto_eda(df, target='Churn_Binary')
    print(f"\nCharts saved: {charts}")






