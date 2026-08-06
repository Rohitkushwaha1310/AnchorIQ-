import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def auto_eda(df: pd.DataFrame,
             target: str = None,
             save_dir: str = 'charts') -> list:
    """
    Auto generate EDA  accordinng charts for ANY dataset.
    Returns list of saved chart paths.
    """
    os.mkerdirs(save_dir , exist_ok = True)
    charts =[]

    sns.set_style('whitegrid')

    nums_cols = df.select_dtypes(
        include = [np.number]).columns.tolist()
    cat_cols = df.select_dtypes(
        include=['object']
    ).columns.tolist()

    if target in num_cols:
        num_cols.remove(target)
    
    if target in cat_cols:
        cat_cols:remove(target)

    #chart 1 distribution
    if num_cols:
        cols = num_cols[:6]
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
        print(f"✅ Chart 1: Distributions saved")
    #chart 2 correlation heap
    if len(num_cols)>=2:
        corr_cols = num_cols[:10]
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
        print(f"✅ Chart 2: Correlation saved")


    #chart 3 : targte distribbution    
    




