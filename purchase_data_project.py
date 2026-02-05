
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)

num_records = 500

tenure_months = np.random.randint(1, 72, num_records)
monthly_charges = np.random.normal(loc=70, scale=20, size=num_records)
monthly_charges = np.clip(monthly_charges, 20, 150)

contract_type = np.random.choice(
    ['Month-to-Month', 'One Year', 'Two Year'],
    size=num_records,
    p=[0.55, 0.25, 0.20]
)

service_tier = np.random.choice(
    ['Basic', 'Standard', 'Premium'],
    size=num_records,
    p=[0.4, 0.35, 0.25]
)


churn_probability = (
    0.45 * (monthly_charges / 150) +
    0.35 * (1 - tenure_months / 72) +
    0.20 * (contract_type == 'Month-to-Month')
)

churn_probability = np.clip(churn_probability, 0, 1)
churn = np.random.binomial(1, churn_probability)


df = pd.DataFrame({
    'TenureMonths': tenure_months,
    'MonthlyCharge': monthly_charges.round(2),
    'ContractType': contract_type,
    'ServiceTier': service_tier,
    'Churn': churn
})

print("\nDataset created successfully with", len(df), "records")

print("\nMissing Values Check:")
print(df.isnull().sum())

df['TenureMonths'] = df['TenureMonths'].astype(int)
df['Churn'] = df['Churn'].astype(int)


print("\nDescriptive Statistics (Numerical Columns):")
print(df[['TenureMonths', 'MonthlyCharge']].describe())

churn_rate = df['Churn'].mean() * 100
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")


churned = df[df['Churn'] == 1]
not_churned = df[df['Churn'] == 0]

avg_monthly_charge_churned = churned['MonthlyCharge'].mean()
avg_monthly_charge_not_churned = not_churned['MonthlyCharge'].mean()

avg_tenure_churned = churned['TenureMonths'].mean()
avg_tenure_not_churned = not_churned['TenureMonths'].mean()

print("\nBivariate Analysis Results:")
print(f"Average Monthly Charge (Churned): {avg_monthly_charge_churned:.2f}")
print(f"Average Monthly Charge (Not Churned): {avg_monthly_charge_not_churned:.2f}")
print(f"Average Tenure Months (Churned): {avg_tenure_churned:.2f}")
print(f"Average Tenure Months (Not Churned): {avg_tenure_not_churned:.2f}")


plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='MonthlyCharge', data=df)
plt.title('Monthly Charge Distribution by Churn Status')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Monthly Charge')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x='ContractType', hue='Churn', data=df)
plt.title('Customer Count by Contract Type and Churn Status')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.legend(title='Churn')
plt.tight_layout()
plt.show()

