import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv("telco_churn.csv")

df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(
    df['TotalCharges'].median(), inplace=True)
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

print(f"✅ Loaded: {df.shape}")

# feature enginnering 

df['ChargesPerTenure'] = (df['TotalCharges']/df['tenure']+1)
df['Isnewcustomer'] = (df['tenure']< 12).astype(int)

service_cols= ['OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport',
                'StreamingTV','StreamingMovies']

df['ServiceCount']= (df[service_cols]== 'Yes').sum(axis=1)   



# high value customer
df['Ishighvalue'] = (
    df['MonthlyCharges'] > df['Monthlycharges'].median()
).astype(int)

#contact risk
contract_map = {
    'Month-to-month' : 3,
    'One year': 2,
    'Two year': 1

}

df[ContractRisk] = df['Contract'].map(contract_map)


# auto pay

df['Isautopay']= df['PaymentMethod'].isin([
    'Bank transfer(automatic)', 
    'Credit card(automatic)'
]).astype(int)