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