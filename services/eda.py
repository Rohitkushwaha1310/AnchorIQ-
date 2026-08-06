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
    os.mkerdirs