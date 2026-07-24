import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("telco_churn.csv")




# print("TELECONNECT INDIA — CHURN ANALYSIS")



# print(f"\nShape          : {df.shape}")
print(f"Columns: {df.columns.tolist()}")
# print(f"\nData Types:\n{df.dtypes}")
# print(f"\nMissing Values:\n{df.isnull().sum()}")
# print(f"\nFirst 5 rows:\n{df.head()}")


# print("CHURN DISTRIBUTION ")
# print(df['Churn'].value_counts())
# print(f"Churn Rate: {(df['Churn']=='Yes').mean()*100:.2f}%") 



 
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)


df['TotalCharges'] = pd.to_numeric (
    df['TotalCharges'], errors='coerce'
)
print(f" tottal changes  after fixing nulls {df['TotalCharges'].isnull(). sum()}")

print("numeric values")
print(df[['tenure','MonthlyCharges',
          'TotalCharges']].describe())

#churning by key features

fig , axes = plt.subplots(2, 3 , figsize = ( 16, 10))
fig.suptitle('Teleconnect india churn EDA', fontsize = 16, fontweight = 'bold')


# chrun distribution

churn_counts = df['Churn'].value_counts()
axes[0,0].pie(churn_counts, labels = churn_counts.index,
                autopct = '%1.1f%%',
                colors = ['#2ecc71','#e74c3c'])

axes[0,0].set_title('Overall churn rate')




# churn by contact type
contract_churn = df.groupby('Contract')['Churn_Binary'].mean()*100
axes[0,1].bar(contract_churn.index,
                contract_churn.values,
                color = ['#e74c3c','#f39c12','#2ecc71'])

axes[0,1].set_title('Churn rate by contract type')
axes[0,1].set_ylabel('churn rate%')
axes[0,1].tick_params(axis='x', rotation=15)


# tensure distrubution yhcurn 

df[df['Churn']=='No']['tenure'].hist(
    ax=axes[0,2], bins=30, alpha=0.6,
    color='green', label='Stayed')

df[df['Churn']=='Yes']['tenure'].hist(
    ax= axes[0,2], bins=30, alpha=0.6,
    color='red', label='Churned'
)    
axes[0,2].set_title('Tenure Distribution by chrun')
axes[0,2].set_xlabel('Tenure(months)')
axes[0,2].legend()

#monthly charges by churn
df.boxplot(column='MonthlyCharges',
             by='Churn', ax=axes[1,0])

axes[1,0].set_title('Mothnly charges by churnn ')             
axes[1,0].set_xlabel('Churn')


# churn by ineret sevices
internet_churn = df.groupby(
    'InternetService')['Churn_Binary'].mean()*100

axes[1,1].bar(internet_churn.index,
              internet_churn.values,
              color = ['#3498db','#e74c3c','#2ecc71'])

axes[1,1].set_title('Churn rate by internet service')
axes[1,1].set_ylabel('Churnn rate %')

# chur by paymenbt 
payment_churn= df.groupby(
    'Payment Method')['Churn_Binary'].mean()*100
axes[1,2].bar(payment_churn.index,
              payment_churn.values,
              color='steelblue')

axes[1,2].set_title('Churn by payment mehtood')
axes[1,2].set_xlabel('CHurn rate %')


# plt.tight_layout()
# plt.show()

# print("\n=== KEY CHURN INSIGHTS ===")
# print(f"Avg tenure (churned)    : {df[df['Churn']=='Yes']['tenure'].mean():.1f} months")
# print(f"Avg tenure (stayed)     : {df[df['Churn']=='No']['tenure'].mean():.1f} months")
# print(f"Avg charges (churned)   : ${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f}")
# print(f"Avg charges (stayed)    : ${df[df['Churn']=='No']['MonthlyCharges'].mean():.2f}")
# print(f"\nChurn by contract:")
# print(df.groupby('Contract')['Churn_Binary'].mean()*100)


df_corr = df.copy()
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df_corr[col] = pd.factorize(df_corr[col])[0]

churn_corr = df_corr.corr()['Churn_Binary'].sort_values(
    ascending =False
)    

print("correleation with churn")

print(churn_corr.to_string())


#visualization of correlation
fig, axes = plt.subplots(1,2 , figsize=(14,6))

top_corr = churn_corr.drop('Churn_Binary').head(10)
colors = ['red' if x > 0 else 'green' for x in top_corr.values]

axes[0].barh(top_corr.index, top_corr.values, color= colors)

axes[0].axvline(0, color ='black', linewidth=0.5)
axes[0].set_title('top Feature correleated with churn')
axes[0].set_xlabel('correlation')


#heatmap fo features
key_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Binary', 'Contract', 'PaymentMethod']
sns.heatmap(
    df_corr[key_features].corr(),
    annot=True, fmt='0.2f',
    cmap='coolwarm', ax= axes[1]
)
axes[1].set_title('feature correlation heatmap')
plt.tight_layout()
plt.show()


#statical test
from scipy import stats

churned = df[df['Churn']=='Yes']['tenure']
stayed =df[df['Churn']=='No']['tenure']

t_stat, p_value = stats.ttest_ind(churned, stayed)

print("statical tes")
print(f"Churned avg tenure : {churned.mean():.2f} months")
print(f"Stayed avg tenure  : {stayed.mean():.2f} months")
print(f"T-statistic        : {t_stat:.4f}")

if p_value < 0.05:
    print("✅ Difference is STATISTICALLY SIGNIFICANT!")
else:
    print("❌ No significant difference")


print("CHURN RISK SEGMENTS")
df['Risk_Segment'] = 'Low Risk'
df.loc[(df['Contract']=='Month-to-month') &
       (df['tenure'] < 12), 'Risk_Segment'] = 'High Risk'
df.loc[(df['Contract']=='Month-to-month') &
       (df['tenure'].between(12,24)),
       'Risk_Segment'] = 'Medium Risk'

print(df['Risk_Segment'].value_counts())
print("\nChurn rate by risk segment:")
print(df.groupby('Risk_Segment')['Churn_Binary'].mean()*100)