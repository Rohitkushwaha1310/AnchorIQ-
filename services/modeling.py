import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def auto_model(df: pd.DataFrame,
               target: str,
               save_dir: str= "models")->dict:

    """
    Auto detect problem type & train best model
    Works with ANY dataset!
    """
    os.makedirs(save_dir, exist_ok=True)   


    #detect problem type
    
    n_unique = df[target].nunique()
    problem_type= 'classification' if n_unique < 10 else 'regression'

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

    #keep onnly numerical
    df_ml = df_ml.select_dtypes(include= [np.number])

    X= df_ml.drop(target, axis =1, errors ='ignore') 
    y= df_ml[target]   

    #remove null
    mask = X.notna().all(axis=1) & y.notna()
    X,y = X[mask], y[mask]

    print(f" feature : {X.shape[1]}")
    print(f" samples : {len(X)}")

    #split
    stratify = y if problem_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size =0.2, random_state =42, stratify = stratify)
    results={}


    if problem_type == 'classification':
        #modle 1 logistic regression
        print("\n🔄 Training Logistic Regression...")

        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=1000, random_state=42))
        ])
        lr_pipe.fit(X_train, y_train)
        lr_pred  = lr_pipe.predict(X_test)
        lr_proba = lr_pipe.predict_proba(X_test)[:,1]
        lr_auc   = roc_auc_score(y_test, lr_proba)
        print(f"   LR AUC: {lr_auc:.4f}")

         # Model 2 — Random Forest
        print("🔄 Training Random Forest...")
        rf_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=100,
                random_state=42, n_jobs=-1))
        ])
        rf_pipe.fit(X_train, y_train)
        rf_pred  = rf_pipe.predict(X_test)
        rf_proba = rf_pipe.predict_proba(X_test)[:,1]
        rf_auc   = roc_auc_score(y_test, rf_proba)
        print(f"   RF AUC: {rf_auc:.4f}")

          # Pick best model
        if lr_auc >= rf_auc:
            best_pipe = lr_pipe
            best_name = 'Logistic Regression'
            best_pred = lr_pred
            best_proba= lr_proba
            best_auc  = lr_auc
        else:
            best_pipe = rf_pipe
            best_name = 'Random Forest'
            best_pred = rf_pred
            best_proba= rf_proba
            best_auc  = rf_auc

           # Cross validation
        cv = cross_val_score(
            best_pipe, X, y,
            cv=5, scoring='roc_auc')

        # Confusion matrix
        cm = confusion_matrix(y_test, best_pred) 

        results = {
            'problem_type'    : problem_type,
            'best_model'      : best_name,
            'accuracy'        : round(
                accuracy_score(y_test, best_pred)*100, 2),
            'auc'             : round(best_auc, 4),
            'lr_auc'          : round(lr_auc, 4),
            'rf_auc'          : round(rf_auc, 4),
            'cv_mean'         : round(cv.mean(), 4),
            'cv_std'          : round(cv.std(), 4),
            'train_samples'   : len(X_train),
            'test_samples'    : len(X_test),
            'n_features'      : X.shape[1],
            'confusion_matrix': cm.tolist(),
            'report'          : classification_report(
                y_test, best_pred, output_dict=True)
        }

        joblib.dump({
            'pipeline': best_pipe,
            'features': list(X.columns),
            'target'  : target,
            'auc'     : best_auc,
            'problem' : problem_type
        }, f'{save_dir}/model.pkl')

        print(f"\n🏆 Best: {best_name} (AUC={best_auc:.4f})")
        print(f"✅ Model saved to {save_dir}/model.pkl")

    return results


# Add at bottom to test
if __name__ == "__main__":
    df = pd.read_csv("telco_churn.csv")
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(
        df['TotalCharges'].median(), inplace=True)

    results = auto_model(df, target='Churn')
    print("\n=== MODEL RESULTS ===")
    print(f"Best model  : {results['best_model']}")
    print(f"AUC Score   : {results['auc']}")
    print(f"Accuracy    : {results['accuracy']}%")
    print(f"CV Mean AUC : {results['cv_mean']}")

