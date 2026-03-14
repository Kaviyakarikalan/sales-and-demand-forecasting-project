# sales_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Ensure plots display properly in scripts
plt.switch_backend('agg')  # avoids multiple plt.show() issues in some environments

# -------------------------
# Step 1: Load dataset
# -------------------------
df = pd.read_csv("amazon_sales.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# -------------------------
# Step 2: Data Cleaning
# -------------------------
target_column = "actual_price"

# Remove currency symbols and convert to numeric
df[target_column] = (
    df[target_column].astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)
df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

# -------------------------
# Step 2.1: Clean feature columns
# -------------------------
features = ["discount_percentage", "rating", "rating_count"]

# Remove % from discount_percentage and convert to float
df["discount_percentage"] = df["discount_percentage"].str.replace("%", "", regex=False)
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")

# Convert rating to numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Remove commas from rating_count and convert to int
df["rating_count"] = df["rating_count"].str.replace(",", "", regex=False)
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

# Drop rows with NaN in features or target
df = df.dropna(subset=features + [target_column])

print("\nDataset after cleaning:")
print(df.head())

# -------------------------
# Step 3: Price Distribution
# -------------------------
plt.figure(figsize=(8,5))
sns.histplot(df[target_column], kde=True, bins=30)
plt.title("Actual Price Distribution")
plt.xlabel("Actual Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.close()

# -------------------------
# Step 4: Category vs Price
# -------------------------
if "category" in df.columns:
    top_categories = (
        df.groupby("category")[target_column]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,5))
    top_categories.plot(kind="bar")
    plt.title("Top Categories by Average Price")
    plt.xlabel("Category")
    plt.ylabel("Average Price")
    plt.tight_layout()
    plt.savefig("top_categories.png")
    plt.close()

# -------------------------
# Step 5: Feature Selection
# -------------------------
X = df[features]
y = df[target_column]

print("\nFeatures after cleaning:")
print(X.head())

print("\nTarget:")
print(y.head())

# -------------------------
# Step 6: Train Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# -------------------------
# Step 7: Linear Regression
# -------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\nLinear Regression Results")
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2:", r2_score(y_test, lr_pred))

# -------------------------
# Step 8: Random Forest
# -------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))

# -------------------------
# Step 9: Error Metrics
# -------------------------
mae = mean_absolute_error(y_test, rf_pred)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\nModel Error Metrics")
print("MAE:", mae)
print("RMSE:", rmse)

# -------------------------
# Step 10: Prediction Plot
# -------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, lr_pred, label="Linear Regression", alpha=0.6)
plt.scatter(y_test, rf_pred, label="Random Forest", alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_scatter.png")
plt.close()

# -------------------------
# Step 11: Forecast Visualization
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(y_test.values[:50], label="Actual Price", marker='o')
plt.plot(rf_pred[:50], label="Predicted Price", marker='x')
plt.title("Sales Forecast Comparison")
plt.xlabel("Products")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_comparison.png")
plt.close()

# -------------------------
# Step 12: Feature Importance
# -------------------------
importance = rf_model.feature_importances_
plt.figure(figsize=(6,4))
plt.bar(features, importance, color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# -------------------------
# Step 13: Save Model
# -------------------------
joblib.dump(rf_model, "sales_forecast_model.pkl")
print("\nModel saved as sales_forecast_model.pkl")

# -------------------------
# Step 14: Example Prediction
# -------------------------
sample = X_test.iloc[[0]]
prediction = rf_model.predict(sample)
print("\nPredicted price for sample product:", prediction)