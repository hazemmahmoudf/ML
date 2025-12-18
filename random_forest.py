import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Data
# =========================
df = pd.read_csv("dataset/cleaned_household_power_data.csv", nrows=100000)

# =========================
# Target & Features
# =========================
target_col = 'Global_active_power'
feature_cols = [
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]

X = df[feature_cols]
y = df[target_col]

# =========================
# Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# Random Forest Regression
# =========================
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# =========================
# Evaluation Metrics
# =========================
print("===== Random Forest Regression Results =====")
print(f"R²    : {r2_score(y_test, y_pred):.4f}")
print(f"MAE   : {mean_absolute_error(y_test, y_pred):.4f} kW")
print(f"MSE   : {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAPE  : {mean_absolute_percentage_error(y_test, y_pred)*100:.2f} %")

# =========================
# Visualization
# =========================

# 1️⃣ Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)
plt.xlabel("Actual Global Active Power (kW)")
plt.ylabel("Predicted Global Active Power (kW)")
plt.title("Random Forest: Actual vs Predicted")
plt.grid(True)
plt.show()

# 2️⃣ Residuals
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.boxplot(x=residuals)
plt.title("Residuals Distribution - Random Forest")
plt.xlabel("Residual (kW)")
plt.show()



# =========================
# Predict New Sample
# =========================
new_data = pd.DataFrame({
    'Global_reactive_power': [0.1],
    'Voltage': [240],
    'Global_intensity': [10],
    'Sub_metering_1': [0],
    'Sub_metering_2': [1],
    'Sub_metering_3': [2]
})

prediction = rf_model.predict(new_data)[0]
print(f"\nPredicted Global Active Power: {prediction:.3f} kW")
