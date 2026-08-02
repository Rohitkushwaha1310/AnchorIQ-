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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
import joblib
import os
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


print("deep evaluation - logistsi regression")

print('classification report')
print(classification_report(y_test, lr_pred,
                target_names = ['Stayed', 'Churned']))

cm= confusion_matrix(y_test, lr_pred)
tn, fp, fn, tp = cm.ravel()

print(" confusion matrix")
print(f"True Negatives  (Stayed correctly)  : {tn}")
print(f"False Positives (Wrong churn flag)   : {fp}")
print(f"False Negatives (Missed churners!)   : {fn}")
print(f"True Positives  (Caught churners!)   : {tp}")


# bussiness costs
cost_fn = 500
cost_fp =  50
total_cost = (fn* cost_fn)+ (fp* cost_fp)
cost_saved = tp*500*0.3


print(f" BUSINESS IMPACT")
print(f"Churners caught      : {tp}")
print(f"Churners missed      : {fn}")
print(f"Cost of missed       : ${fn*cost_fn:,}")
print(f"Cost of false alarms : ${fp*cost_fp:,}")
print(f"Total cost           : ${total_cost:,}")
print(f"Est. revenue saved   : ${cost_saved:,.0f}")

# cross validation 
print( " 5 fold cross validation")

cv_scores = cross_val_score(
    lr, X_train_scaled, y_train, cv=5, scoring='roc_auc'
)
print(f"CV AUC Scores : {cv_scores.round(4)}")
print(f"Mean AUC      : {cv_scores.mean():.4f}")
print(f"Std Dev       : {cv_scores.std():.4f}")

#visaulization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('AnchorIQ — Model Evaluation', fontsize=16)

#heatmap of confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed','Churned'],
            yticklabels=['Stayed','Churned'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')


# roc curve of all three models
for name, (pred, proba), color in zip(
    ['Logistic Reg','Random Forest','XGBoost'],
    [(lr_pred, lr_proba),(rf_pred, rf_proba),(xgb_pred, xgb_proba)],
    ['blue','green','red']):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_score = roc_auc_score(y_test, proba)
    axes[1].plot(fpr, tpr, color=color,
                 label=f'{name} (AUC={auc_score:.3f})')
axes[1].plot([0,1],[0,1],'k--', label='Random')
axes[1].set_title('ROC Curve — All Models')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()


#cv score
axes[2].bar(range(1,6), cv_scores,
            color='steelblue', edgecolor='white')
axes[2].axhline(cv_scores.mean(), color='red',
                linestyle='--',
                label=f'Mean={cv_scores.mean():.3f}')
axes[2].set_title('Cross Validation AUC Scores')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('AUC Score')
axes[2].legend()
axes[2].set_ylim(0.7, 1.0)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Evaluation charts saved!")

#thresold tuning
print("thresold tuning for busineess")

thresholds = np.arange(0.1, 0.9, 0.05)
results_thresh=[]

for thresh in thresholds:
    pred= (lr_proba >= thresh).astype(int)
    cm_t = confusion_matrix(y_test, pred)
    tn_t, fp_t, fn_t, tp_t= cm_t.ravel()

    cost = (fn_t*500)+ (fp_t *50)
    saved = tp_t*500*0.3
    rec = recall_score(y_test, pred)
    prec= recall_score(y_test, pred)
    
    f1 = f1_score(y_test, pred, zero_division=0)


    results_thresh.append({
         'Threshold' : round(thresh, 2),
        'Recall'    : round(rec, 4),
        'Precision' : round(prec, 4),
        'F1'        : round(f1, 4),
        'Cost'      : cost,
        'Saved'     : round(saved, 0),
        'TP'        : tp_t,
        'FN'        : fn_t
    })


thresh_df = pd.DataFrame (results_thresh)  


print(thresh_df[['Threshold','Recall','Precision','F1','Cost','Saved']].to_string(index=False))

# best threshold by minimum cost

best_idx = thresh_df['Cost'].idxmin()
best_thresh = thresh_df.loc[best_idx, 'Threshold']
best_cost = thresh_df.loc[best_idx, 'Cost']
best_saved = thresh_df.loc[best_idx, 'Saved']



print(f"\n🎯 Optimal Threshold : {best_thresh}")
print(f"   Minimum Cost     : ${best_cost:,}")
print(f"   Revenue Saved    : ${best_saved:,}")


#visualization

fig, axes = plt.subplots(1,2, figsize=(14,5))

#cost curve

axes[0].plot(thresh_df['Threshold'],
             thresh_df['Cost'],
             'r-o', linewidth=2)
axes[0].axvline(best_thresh, color='green',
                linestyle='--',
                label=f'Optimal={best_thresh}')
axes[0].set_title('Business Cost vs Threshold')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Total Cost ($)')
axes[0].legend()


#recall vs preccision
axes[1].plot(thresh_df['Threshold'],
             thresh_df['Recall'],
             'b-o', label='Recall')
axes[1].plot(thresh_df['Threshold'],
             thresh_df['Precision'],
             'g-o', label='Precision')
axes[1].plot(thresh_df['Threshold'],
             thresh_df['F1'],
             'r-o', label='F1')
axes[1].axvline(best_thresh, color='black',
                linestyle='--',
                label=f'Optimal={best_thresh}')
axes[1].set_title('Metrics vs Threshold')
axes[1].set_xlabel('Threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig('threshold_tuning.png',
            dpi=150, bbox_inches='tight')
plt.show()


print("saving model")

model_data = {
    'model'    : lr,
    'scaler'   : scaler,
    'features' : list(X_train.columns),
    'threshold': best_thresh,
    'auc'      : cv_scores.mean()

}

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

joblib.dump(model_data, 'models/churn_model.pkl')
size = os.path.getsize('models/churn_model.pkl')/1024
print(f"MOdel saved size : {size:.2f} KB")


#load and verify
loaded= joblib.load('models/churn_model.pkl')
model_loaded = loaded['model']
scaler_loaded = loaded['scaler']
threshold_loaded = loaded['threshold']
features_loaded = loaded['features']

print(f"loaded  and verify ")
print(f"   Threshold : {threshold_loaded}")
print(f"   AUC       : {loaded['auc']:.4f}")
print(f"   Features  : {len(features_loaded)}")

#predict new customer

#simulate f5 customer comming inn 
new_customers = pd.DataFrame({
    'tenure'          : [2,  45,  1, 60,  8],
    'MonthlyCharges'  : [85, 45, 95, 35, 78],
    'TotalCharges'    : [170, 2025, 95, 2100, 624],
    'SeniorCitizen'   : [0, 0, 1, 0, 0],
    'ContractRisk'    : [3, 2, 3, 1, 3],
    'Isautopay'       : [0, 1, 0, 1, 0],
    'Isnewcustomer'   : [1, 0, 1, 0, 1],
    'ServiceCount'    : [1, 4, 0, 5, 2],
    'IshighValue'     : [1, 0, 1, 0, 1],
    'ChargesPerTenure': [56.7, 44.7, 47.5, 34.4, 74.7]
})

new_customers = new_customers.reindex(columns= features_loaded, fill_value=0)


#sales and predict
new_scaled = scaler_loaded.transform(new_customers)
new_scaled = scaler_loaded.transform(new_customers)
probabilities = model_loaded.predict_proba(new_scaled)[:,1]
predictions   = (probabilities >= threshold_loaded).astype(int)

print("new customer prediction")
pred_df = pd.DataFrame({
    'Customer'   : ['C1','C2','C3','C4','C5'],
    'Tenure'     : [2, 45, 1, 60, 8],
    'Monthly$'   : [85, 45, 95, 35, 78],
    'Churn Risk' : [f"{p*100:.1f}%" for p in probabilities],
    'Prediction' : ['🚨 WILL CHURN' if p==1
                    else '✅ WILL STAY' for p in predictions],
    'Action'     : ['Send retention offer!' if p==1
                    else 'No action needed' for p in predictions]
})

print(pred_df.to_string(index=False))

#save preddcition
pred_df.to_csv('reports/predictions.csv', index=False)
print("\n✅ Predictions saved!")
print("\n🏆 churn_model.py COMPLETE!")


    


    