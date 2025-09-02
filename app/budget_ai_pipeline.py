import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Config
# -------------------------------
RANDOM_STATE = 42
BUDGET_COLS = [
    "Savings_Amount", "Groceries_Spending", "Transport_Spending",
    "Entertainment_Spending", "Utilities_Spending", "Loan_Repayment",
    "Insurance", "Shopping_Spending", "Investment_Amount", "Other_Spending"
]

# -------------------------------
# 1. Feature engineering
# -------------------------------
def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Spending_Ratio"] = np.where(df["Monthly_Income"] > 0,
        df["Total_Spending"] / df["Monthly_Income"], np.nan)
    df["Savings_Ratio"] = np.where(df["Monthly_Income"] > 0,
        df["Savings_Amount"] / df["Monthly_Income"], np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=["Spending_Ratio", "Savings_Ratio"])

# -------------------------------
# 2. Clustering + labeling
# -------------------------------
def cluster_users(df: pd.DataFrame, k: int = 3):
    X = df[["Spending_Ratio", "Savings_Ratio"]]
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
    labels = kmeans.fit_predict(X)

    cent = pd.DataFrame(kmeans.cluster_centers_, columns=["Spending_Ratio","Savings_Ratio"])
    order = cent["Savings_Ratio"].rank(method="first").astype(int)

    id_to_label = {}
    for cid in cent.index:
        if order[cid] == k: id_to_label[cid] = "Tiết kiệm"
        elif order[cid] == 1: id_to_label[cid] = "Chi tiêu cao"
        else: id_to_label[cid] = "Trung bình"

    df = df.copy()
    df["Cluster"] = labels
    df["Cluster_Label"] = df["Cluster"].map(id_to_label)

    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    return df, kmeans, id_to_label, sil, dbi

# -------------------------------
# 3. Train budget model
# -------------------------------
def train_budget_model(df: pd.DataFrame):
    df = df.copy()
    for col in BUDGET_COLS:
        df[col + "_Rate"] = (df[col] / df["Monthly_Income"] * 100)

    budget_cols_rate = [c + "_Rate" for c in BUDGET_COLS]

    X = df.drop(columns=BUDGET_COLS + ["Cluster","Cluster_Label","Goal_Amount","Total_Spending"], errors="ignore")
    y = df[budget_cols_rate]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print("Train feature: ",X_tr.columns.tolist())
    model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=2))
    model.fit(X_tr, y_tr)

    pred = pd.DataFrame(model.predict(X_te), columns=budget_cols_rate, index=y_te.index)
    pred = enforce_budget_constraints(pred)

    mae = (pred - y_te).abs().mean()
    rmse = np.sqrt(((pred - y_te)**2).mean())
    return model, budget_cols_rate, mae, rmse

# -------------------------------
# 4. Post-processing constraints
# -------------------------------
def enforce_budget_constraints(df: pd.DataFrame):
    df = df.clip(lower=0)
    row_sums = df.sum(axis=1).replace(0, 1e-6)
    return df.div(row_sums, axis=0) * 100

# -------------------------------
# 5. Recommendation API
# -------------------------------
# def recommend(model, X_row: pd.Series, budget_cols_rate, known_rates: dict):
#     pred = pd.Series(model.predict(X_row.to_frame().T)[0], index=budget_cols_rate)
#     pred = pred.clip(lower=0)

#     remaining = 100 - sum(known_rates.values())
#     remaining = max(remaining, 0)

#     unknown = [c for c in budget_cols_rate if c not in known_rates]
#     scaled_unknown = pred[unknown]
#     s = scaled_unknown.sum()
#     if s <= 0: scaled_unknown[:] = remaining / len(unknown)
#     else: scaled_unknown = scaled_unknown * (remaining / s)

#     final = pd.Series(0.0, index=budget_cols_rate)
#     for k,v in known_rates.items(): final[k] = v
#     final[unknown] = scaled_unknown
#     return final.round(2)
def recommend(model, X_row, budget_cols_rate, known_rates):
    """
    X_row: dict with user inputs (numbers in VND)
    budget_cols_rate: list of ratio feature names used by the model
    known_rates: dict of any known ratios (if provided directly)
    """
    

    total_income = X_row["Monthly_Income"]
    
    # Step 1: Convert numeric budgets to ratios
    ratios = {}
    for col in budget_cols_rate:
        base_name = col.replace("_Ratio", "")
        if base_name in X_row:
            ratios[col] = X_row[base_name] / total_income

    # Step 2: Merge with known ratios
    ratios.update(known_rates)

    # Step 3: Prepare DataFrame with all required features
    # X_row_df = pd.DataFrame([{col: ratios.get(col, 0) for col in budget_cols_rate}])
    # X_row_df = pd.DataFrame([X_row])
    X_row_df = pd.DataFrame([{**X_row,**{col: ratios.get(col, 0) for col in budget_cols_rate}}])

    print("Input of pipeline: \n", X_row_df.columns.tolist()) #debug
    # Step 4: Predict missing ratios
    pred = pd.Series(model.predict(X_row_df)[0], index=budget_cols_rate)
    print("Prediction: ", pred)
    # Step 5: Merge predictions with known ratios
    # plan_rates = pred.to_dict()
    # plan_rates = pd.Series(pred, index=budget_cols_rate)

    # plan_rates.update(ratios)
    
    # # Step 6: Convert back to absolute amounts
    # plan_amounts = {col.replace("_Ratio", ""): rate * total_income for col, rate in plan_rates.items()}

    # # Step 7: Auto-calc Other_spending
    # if "Other_spending" in plan_amounts:
    #     spent_without_other = sum(v for k, v in plan_amounts.items() if k != "Other_spending")
    #     plan_amounts["Other_spending"] = total_income - spent_without_other
    #     plan_rates["Other_spending_Ratio"] = plan_amounts["Other_spending"] / total_income
    return pred
df = pd.read_csv("synthetic_financial_behavior_dataset_balanced_rounded.csv")

# Feature engineering + clustering
df = add_ratios(df)
df, km, id2label, sil, dbi = cluster_users(df, k=3)

# Train model
model, budget_cols_rate, mae, rmse = train_budget_model(df)

# Save model + metadata
joblib.dump(model, "budget_model.pkl")
joblib.dump(budget_cols_rate, "budget_cols_rate.pkl")
joblib.dump((mae, rmse), "metrics.pkl")

print("✅ Model saved successfully!")