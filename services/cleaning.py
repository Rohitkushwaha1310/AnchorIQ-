import pandas as pd
import numpy as np

def auto_clean(df:pd.DataFrame):
    """ auto clean any dataset.
    return : (cleaned_df, clenaing_report)
    """
    report= {
        'orginal_shape': df.shape,
        'step_applied': []
    }


    #step 1; drop duplicates
    before = len(df)
    df= df.drop_duplicates()
    removed = before -len(df)
    if removed >0:
        report['step_applied'].append(f"Removed{removed} duplicat rows")



    #step 2 fix numeric stores as string

    for col in df.columns:
        if df[col].dtype == 'object':
            converted = pd.to_numeric(
                df[col], errors ='coerce'
            )
            if converted.notna().mean()> 0.8:
                df[col]= converted
                report['step_applied'].append(f"Converted {col} to numeric")



    #step 3 fill missing values
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count>0:
            if df[col].dtype in ['float64', 'int64']:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True) 
                report['step_applied'].append(f"filled {null_count} nulls in {col} with median {fill_val:.2f}")       
            else:
                fill_val = df[col].mode()[0] if len(df[col].mode())>0 else 'Unknown'
                df[col].fillna(fill_val, inplace = True)
                report['step_applied'].append(f"filled {null_count} nulls in {col} with mode {fill_val}")

    
    
    
    
    
    #step 4 strip whitespace
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col]= df[col].str.strip()

    if len(str_cols)>0:
        report['step_applied'].append(f"Stripped whitespace from {len(str_cols)} text columns")


    # Step 5 — Fix date columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(
                    df[col], errors='coerce')
                report['step_applied'].append(
                    f"Converted '{col}' to datetime")
            except:
                pass    
    

    report['cleaned_shape']= df.shape
    report['nulls_remaining']= int(df.isnull().sum().sum())


    return df, report


if __name__ == "__main__":
    df = pd.read_csv("telco_churn.csv")
    df_clean, report = auto_clean(df)

    print("=== CLEANING REPORT ===")
    print(f"Original shape : {report['orginal_shape']}")
    print(f"Cleaned shape  : {report['cleaned_shape']}")
    print(f"Nulls remaining: {report['nulls_remaining']}")
    print(f"\nSteps applied:")
    for step in report['step_applied']:
        print(f"  ✅ {step}")
