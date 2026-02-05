import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n = 600

customer_id = np.arange(1, n + 1)
age = np.random.randint(18, 65, n)

income = np.random.normal(loc=50000, scale=15000, size=n)
income = np.clip(income, 15000, 120000)

product_category = np.random.choice(
    ['Electronics', 'Clothing', 'Groceries', 'Sports'],
    size=n
)

purchase_amount = income * 0.05 + np.random.normal(200, 80, n)
purchase_amount = np.clip(purchase_amount, 50, 2000)

df = pd.DataFrame({
    "Customer_ID": customer_id,
    "Age": age,
    "Income": income.round(2),
    "Purchase_Amount": purchase_amount.round(2),
    "Product_Category": product_category
})

print("\n================ Dataset Preview ================\n")
print(df.head())

print("\n================ Missing Values ================\n")
print(df.isnull().sum())

df["Customer_ID"] = df["Customer_ID"].astype(int)
df["Age"] = df["Age"].astype(int)
df["Product_Category"] = df["Product_Category"].astype("category")

print("\n================ df.describe() Output ================\n")
print(df.describe())

print("\n================ Manual Statistics ================\n")

for col in ["Income", "Purchase_Amount"]:
    print(f"\n----- {col} -----")
    print("Mean   :", df[col].mean())
    print("Median :", df[col].median())
    print("Std    :", df[col].std())
    print("Min    :", df[col].min())
    print("Max    :", df[col].max())

plt.figure(figsize=(7,5))
plt.hist(df["Income"], bins=25)
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Income Distribution Histogram")
plt.tight_layout()
plt.savefig("income_histogram.png")
plt.show()

plt.figure(figsize=(7,5))

categories = df["Product_Category"].unique()
data = [
    df[df["Product_Category"] == cat]["Purchase_Amount"]
    for cat in categories
]

plt.boxplot(data, labels=categories)
plt.xlabel("Product Category")
plt.ylabel("Purchase Amount")
plt.title("Purchase Amount by Product Category")
plt.tight_layout()
plt.savefig("category_boxplot.png")
plt.show()

print("\n================ Key Findings ================\n")

avg_income = df["Income"].mean()
avg_purchase = df["Purchase_Amount"].mean()

print(f"Average customer income is {avg_income:.2f}")
print(f"Average purchase amount is {avg_purchase:.2f}")

category_avg = df.groupby("Product_Category")["Purchase_Amount"].mean()

print("\nAverage purchase by category:")
print(category_avg)

highest_cat = category_avg.idxmax()

print(f"\nCustomers spend the most in '{highest_cat}' category.")

print("\nHistogram and boxplot images saved as:")
print("  -> income_histogram.png")
print("  -> category_boxplot.png")

print("\n================ Project Completed Successfully ================")