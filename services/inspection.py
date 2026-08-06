import pandas as pd
import numpy as np

def inspect_dataset(df:pd.DataFrame):
    n_rows, n_cols = df.shape

    numerical_cols = df.select_dtypes(include= [np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include = ['object']).columns.tolist()


    missing_per_col = df.isnull().sum()
    missing_total = int (missing_per_col.sum())
    missing_cols = missing_per_col[missing_per_col>0].to_dict()
    duplicates = int(df.duplicated().sum())
    target = _guess_target(df)


    stats = {}
    for col in numerical_cols[:8]:
        stats[col]={
            'mean'    : round(float(df[col].mean()), 2),
            'median'  : round(float(df[col].median()), 2),
            'std'     : round(float(df[col].std()), 2),
            'min'     : round(float(df[col].min()), 2),
            'max'     : round(float(df[col].max()), 2),
            'skewness': round(float(df[col].skew()), 2),
            'nulls'   : int(df[col].isnull().sum())

        }

        return{
            'rows'            : n_rows,
        'columns'         : n_cols,
        'numerical_cols'  : numerical_cols,
        'categorical_cols': categorical_cols,
        'missing_total'   : missing_total,
        'missing_cols'    : missing_cols,
        'duplicates'      : duplicates,
        'target_column'   : target,
        'stats'           : stats
        }


def _guess_target(df:pd.DataFrame):
    common = [
        'churn','target','label','class','outcome',
        'result','status','fraud','converted',
        'default','survived','purchased','clicked'
    ]

    lower_cols = {c.lower():c for c in df.columns}
    for name in common :
        if name in lower_cols:
            return lower_cols[name]
    for col in reversed(df.columns):
        if 2<=df[col].nunique(dropna=True)<=10: 
            return col       
    return None
   
    

if __name__ == "__main__":
    df = pd.read_csv("telco_churn.csv")
    result = inspect_dataset(df)
    print("=== INSPECTION RESULT ===")
    print(f"Rows           : {result['rows']}")
    print(f"Columns        : {result['columns']}")
    print(f"Target detected: {result['target_column']}")
    print(f"Numerical cols : {result['numerical_cols']}")
    print(f"Missing total  : {result['missing_total']}")
    print(f"Duplicates     : {result['duplicates']}")    
    
  