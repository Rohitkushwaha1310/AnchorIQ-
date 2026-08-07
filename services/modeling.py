import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from skleanr.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split,
                                      cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score,
                              accuracy_score,
                              classification_report,
                              confusion_matrix)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def auto_model(df: DataFrame,
               target: str,
               save_dir: str= "models")->dict:

    """
    Auto detect problem type & train best model
    Works with ANY dataset!
    """
    os.makedirs(save_dir, exist_ok=True)   


    #detect problem type
    
    n_uique = df[target].nunique()
    problem_type= 'classification' if n_uique < 10 else 'regression'

    print(f" Problem type : {problem_type}")
    print(f"   Target       : {target}")
    print(f"   Unique values: {n_unique}")


    #peapre data 
    df_ml = df.copy()

    #encode target if exist
    if df_ml[target].dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df_ml[target] = le.fit_transform(
            df_ml[target].astype(str))

    cat_cols = df_ml.select_dtypes(
        include ='object').columns.tolist()
    if cat_cols:
        df_ml = pd.get_dummies(
            df_ml, columns=cat_cols, drop_first=True)    
     
