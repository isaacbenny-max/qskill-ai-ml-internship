"""
QSkill AI/ML Internship - Task 3: House Price Prediction (Linear Regression)

DATA: by default this uses the California Housing dataset from scikit-learn
(the Boston dataset was removed from scikit-learn because of ethical concerns).
If the dataset cannot be downloaded (no internet), it falls back to a
synthetic housing dataset with size, bedrooms, age and location features so the
pipeline still runs end to end.
You can also point to your own CSV: set CSV_PATH and TARGET below.
"""
import numpy as np
import os
os.makedirs("outputs", exist_ok=True)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CSV_PATH, TARGET = None, "price"      # e.g. CSV_PATH="housing.csv"

def load_data():
    if CSV_PATH:
        return pd.read_csv(CSV_PATH), TARGET, "custom CSV"
    try:
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        return d.frame, "MedHouseVal", "California Housing (scikit-learn)"
    except Exception:
        rng = np.random.default_rng(42)
        n = 1500
        df = pd.DataFrame({
            "area_sqft": rng.normal(1600, 450, n).clip(500, 4000),
            "bedrooms": rng.integers(1, 6, n),
            "bathrooms": rng.integers(1, 4, n),
            "age_years": rng.integers(0, 50, n),
            "dist_to_city_km": rng.uniform(1, 40, n),
        })
        df["price"] = (df.area_sqft * 3500 + df.bedrooms * 150000 + df.bathrooms * 200000
                       - df.age_years * 8000 - df.dist_to_city_km * 25000
                       + rng.normal(0, 400000, n) + 1_500_000)
        # introduce a few missing values to demonstrate cleaning
        df.loc[rng.choice(n, 25, replace=False), "area_sqft"] = np.nan
        return df, "price", "SYNTHETIC housing data (fallback)"

df, target, source = load_data()
print("Data source:", source)
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe().round(2))

# 1. Explore distributions -----------------------------------------------------
df.hist(figsize=(12, 8), bins=30)
plt.tight_layout(); plt.savefig("outputs/house_distributions.png", dpi=120); plt.close()
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.tight_layout(); plt.savefig("outputs/house_correlation.png", dpi=120); plt.close()

# 2. Handle missing data ---------------------------------------------------------
df = df.fillna(df.median(numeric_only=True))

# 3. Split, then normalise (fit scaler on train only to avoid leakage) ----------
X, y = df.drop(columns=[target]), df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# 4. Train Linear Regression -------------------------------------------------------
model = LinearRegression().fit(X_train_s, y_train)
pred = model.predict(X_test_s)

# 5. Evaluate -------------------------------------------------------------------------
mse = mean_squared_error(y_test, pred)
print(f"\nMSE  : {mse:,.4f}")
print(f"RMSE : {np.sqrt(mse):,.4f}")
print(f"MAE  : {mean_absolute_error(y_test, pred):,.4f}")
print(f"R^2  : {r2_score(y_test, pred):.4f}")

coef = pd.Series(model.coef_, index=X.columns).sort_values()
print("\nFeature coefficients (standardised):\n", coef.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y_test, pred, alpha=0.4)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax[0].plot(lims, lims, "r--"); ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted")
ax[0].set_title("Actual vs Predicted")
coef.plot.barh(ax=ax[1], title="Feature influence")
plt.tight_layout(); plt.savefig("outputs/house_results.png", dpi=120); plt.close()
