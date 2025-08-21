import pandas as pd
import numpy as np

# Set reproducibility
np.random.seed(42)

# Number of samples
n = 10000

# Generate synthetic financial data
monthly_income = np.random.randint(8_000_000, 25_000_000, size=n)

# Spending categories as fractions of income
groceries = (monthly_income * np.random.uniform(0.08, 0.15, size=n)).astype(int)
transport = (monthly_income * np.random.uniform(0.04, 0.08, size=n)).astype(int)
entertainment = (monthly_income * np.random.uniform(0.03, 0.07, size=n)).astype(int)
utilities = (monthly_income * np.random.uniform(0.03, 0.07, size=n)).astype(int)
loan_repayment = (monthly_income * np.random.uniform(0.05, 0.12, size=n)).astype(int)
insurance = (monthly_income * np.random.uniform(0.02, 0.06, size=n)).astype(int)
shopping = (monthly_income * np.random.uniform(0.04, 0.10, size=n)).astype(int)
investment = (monthly_income * np.random.uniform(0.05, 0.12, size=n)).astype(int)

# Total spending = sum of categories
total_spending = groceries + transport + entertainment + utilities + loan_repayment + insurance + shopping

# Savings = income - (total spending + investment)
savings = monthly_income - (total_spending + investment)

# Some savings may be negative (overspending), set min=0
savings = np.maximum(savings, 0)

# Random financial goals (100M or 200M)
goal_amount = np.random.choice([100_000_000, 200_000_000], size=n)

# Assemble dataset
data = pd.DataFrame({
    "Monthly_Income": monthly_income,
    "Total_Spending": total_spending,
    "Savings_Amount": savings,
    "Groceries_Spending": groceries,
    "Transport_Spending": transport,
    "Entertainment_Spending": entertainment,
    "Utilities_Spending": utilities,
    "Loan_Repayment": loan_repayment,
    "Insurance": insurance,
    "Shopping_Spending": shopping,
    "Investment_Amount": investment,
    "Goal_Amount": goal_amount
})

# Save to CSV
data.to_csv("synthetic_financial_behavior_dataset.csv", index=False)

print("Dataset created → synthetic_financial_behavior_dataset.csv")
print(data.head())
