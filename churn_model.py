import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              f1_score, precision_score, recall_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

df= pd.read_csv("telco_churn.csv")

df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(
    df['TotalCharges'].median(), inplace=True)
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

print(f"✅ Loaded: {df.shape}")

# feature enginnering 

df['ChargesPerTenure'] = df['TotalCharges']/(df['tenure']+1)
df['Isnewcustomer'] = (df['tenure']< 12).astype(int)

service_cols= ['OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport',
                'StreamingTV','StreamingMovies']

df['ServiceCount']= (df[service_cols]== 'Yes').sum(axis=1)   



# high value customer
df['Ishighvalue'] = (
    df['MonthlyCharges'] > df['MonthlyCharges'].median()
).astype(int)

#contact risk
contract_map = {
    'Month-to-month' : 3,
    'One year': 2,
    'Two year': 1

}

df['ContractRisk'] = df['Contract'].map(contract_map)


# auto pay

df['Isautopay']= df['PaymentMethod'].isin([
    'Bank transfer (automatic)', 
    'Credit card (automatic)'
]).astype(int)



print("NEW FEATURES CHURN RATES")
print(f"New customer churn  : {df.groupby('Isnewcustomer')['Churn_Binary'].mean()[1]*100:.1f}%")
print(f"Auto pay churn      : {df.groupby('Isautopay')['Churn_Binary'].mean()[1]*100:.1f}%")
print(f"Manual pay churn    : {df.groupby('Isautopay')['Churn_Binary'].mean()[0]*100:.1f}%")
print(f"High value churn    : {df.groupby('Ishighvalue')['Churn_Binary'].mean()[1]*100:.1f}%")
print(f"Final shape         : {df.shape}")

#visualisationn 
fig, axes = plt.subplots(2,3, figsize =(12, 6))
fig.suptitle('AnchorIQ feature anaylysis', fontsize =16)

new_churn = df.groupby('Isnewcustomer')['Churn_Binary'].mean()*100
axes[0,0].bar(['Existing','New Customer'],
               new_churn.values,
               color=['#2ecc71','#e74c3c'])

axes[0,0].bar(['Existing','New Customer'],
               new_churn.values,
               color=['#2ecc71','#e74c3c'])

for i,v in enumerate(new_churn.values):
    axes[0,0].text(i , v+0.5, f'{v:.1f}%',
       ha='center', fontweight='bold')


#contract type
risk_churn = df.groupby('Contract')['Churn_Binary'].mean()*100
axes[0,1].bar(risk_churn.index, risk_churn.values,
               color=['#e74c3c','#f39c12','#2ecc71'])
axes[0,1].set_title('Churn by Contract')
axes[0,1].tick_params(axis='x', rotation=15)


#service count 
service_churn = df.groupby('ServiceCount')['Churn_Binary'].mean()*100
axes[0,2].bar(service_churn.index, service_churn.values,
               color='steelblue')
axes[0,2].set_title('Churn by Service Count')
axes[0,2].set_xlabel('Services')


#autopay

pay_churn = df.groupby('Isautopay')['Churn_Binary'].mean()*100
axes[1,0].bar(['Manual Pay','Auto Pay'],
               pay_churn.values,
               color=['#e74c3c','#2ecc71'])
axes[1,0].set_title('Churn: Auto vs Manual Pay')
for i, v in enumerate(pay_churn.values):
    axes[1,0].text(i, v+0.5, f'{v:.1f}%',
                   ha='center', fontweight='bold')



#high value
hv_churn = df.groupby('Ishighvalue')['Churn_Binary'].mean()*100
axes[1,1].bar(['Low Value','High Value'],
               hv_churn.values,
               color=['#3498db','#e74c3c'])
axes[1,1].set_title('Churn by Customer Value')
for i, v in enumerate(hv_churn.values):
    axes[1,1].text(i, v+0.5, f'{v:.1f}%',
                   ha='center', fontweight='bold')



#charges per tenure
df[df['Churn_Binary']==0]['ChargesPerTenure'].hist(
    ax=axes[1,2], bins=30, alpha=0.6,
    color='green', label='Stayed')
df[df['Churn_Binary']==1]['ChargesPerTenure'].hist(
    ax=axes[1,2], bins=30, alpha=0.6,
    color='red', label='Churned')
axes[1,2].set_title('Charges Per Tenure')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('feature_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Charts saved!")


print ( " preparinng data for ml")
original_features = [
     'tenure', 'MonthlyCharges', 'TotalCharges',
    'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


engineered_features = [
    'ChargesPerTenure', 'Isnewcustomer',
    'ServiceCount', 'Ishighvalue',
    'ContractRisk', 'Isautopay'
]


all_features  = original_features + engineered_features
target = 'Churn_Binary'

df_ml = df[all_features+ [target]].copy()
cat_cols = df_ml.select_dtypes(include='object').columns.tolist()
print(f"encoding{len(cat_cols)} categorical columns: ")

df_ml = pd.get_dummies(df_ml,
                        columns= cat_cols,
                        drop_first=True)


print(f"\nShape after encoding : {df_ml.shape}")
print(f"Total features       : {df_ml.shape[1]-1}")


X = df_ml.drop(target, axis=1)
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42,
    stratify=y)  # stratify keeps churn ratio same!

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print(f"Churn in train   : {y_train.mean()*100:.1f}%")
print(f"Churn in test    : {y_test.mean()*100:.1f}%")

print("\n✅ Data ready for modeling!")


print("buildinng churn prediction model")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#mdoel 1 logistic regression 
print("training loigtstic regresssion")
lr= LogisticRegression(max_iter = 1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:,1]


#model 2 random forestation 

print("ranndom foreststation")

rf= RandomForestClassifier(
    n_estimators = 1000,
    random_state= 42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]

#model 3 xgboost

print("training XGboost")
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate= 0.1,
    max_depth = 5,
    random_state= 42,
    eval_metric='logloss',
    verbosity=0
)

xgb.fit(X_train, y_train)
xgb_pred= xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:,1]


print("camparision of all models")
print(f"{'Model':<25} {'Accuracy':>10} {'AUC':>8} {'F1':>8} {'Recall':>8}")
print("="*60)



models = {
    'Logistic Regression': (lr_pred, lr_proba),
    'Random Forest'      : (rf_pred, rf_proba),
    'XGBoost'            : (xgb_pred, xgb_proba)
}


results = {}
for name, (pred, proba) in models.items():
    acc = accuracy_score(y_test, pred)*100
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    rec = recall_score(y_test, pred)
    results[name] = {'Accuracy':acc, 'AUC': auc, 'F1':f1, 'Recall': rec}
    print(f"{name:<25} {acc:>10.2f}% {auc:>8.4f} {f1:>8.4f} {rec:>8.4f}")


best_model_name = max(results, key=lambda x: results[x]['AUC'])
print(f"\n Best Model of all that i tried so far : {best_model_name}")
print(f"   AUC Score : {results[best_model_name]['AUC']:.4f}")    

