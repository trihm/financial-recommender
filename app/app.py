import streamlit as st
import pandas as pd
import joblib
from budget_ai_pipeline import recommend

# Load pre-trained model + metadata
model = joblib.load("budget_model.pkl")
budget_cols_rate = joblib.load("budget_cols_rate.pkl")
mae, rmse = joblib.load("metrics.pkl")

st.set_page_config(page_title="AI Financial Planner", layout="centered")
st.title("💰 AI Financial Planner")
st.write("Enter your income and any known budget amounts. The AI will suggest a full personalized financial plan.")

# --- User Input ---
income = st.number_input("Monthly Income (₫)", min_value=1, value=15000000, step=500000)

known_amounts = {}
st.subheader("Optional: Enter known spending amounts (₫)")
for col in budget_cols_rate:
    if col == "Savings_Amount_Rate":
        label = "Savings"
    else:
        label = col.replace("_Rate", "")
    val = st.number_input(f"{label} (₫)", min_value=0, value=0, step=500000)
    if val > 0:
        known_amounts[col] = val

# Convert amounts → rates
known_rates = {}
for col, amount in known_amounts.items():
    rate = (amount / income) * 100
    known_rates[col] = rate

# --- Recommendation ---
if st.button("Generate Plan"):
    total_spending = income - known_amounts.get("Savings_Amount_Rate", 0)
    savings_amount = known_amounts.get("Savings_Amount_Rate", 0)
    X_row = {
        "Monthly_Income": income,
        "Spending_Ratio": total_spending / income if income > 0 else 0,
        "Savings_Ratio": savings_amount / income if income > 0 else 0,
    }
    print("application input: ", X_row)
    # Predict plan
    plan_rates = recommend(model, X_row, budget_cols_rate, known_rates)
    
    # Convert predicted rates → amounts
    plan_amounts = {col: (plan_rates[col] / 100) * income for col in plan_rates.index}

    # Apply user overrides
    for col, amount in known_amounts.items():
        plan_amounts[col] = amount

    # --- Calculate Other_Spending ---
    spent_so_far = sum(amount for col, amount in plan_amounts.items() if col != "Savings_Amount_Rate")
    other_spending = income - spent_so_far
    plan_amounts["Other_Spending"] = max(other_spending, 0)  # avoid negative

    # Convert back to percentages
    plan_rates = {col: (val / income) * 100 for col, val in plan_amounts.items()}

    # Display
    st.subheader("📊 Recommended Budget Plan")
    st.write("**Percentages (%)**")
    st.dataframe(pd.Series(plan_rates, name="Percentage (%)")) # st.dataframe(plan_rates)
    st.write("**Amounts (₫)**")
    st.dataframe(pd.Series(plan_amounts, name="Amount (₫)")) # st.dataframe(plan_amounts)