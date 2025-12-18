import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ===== Load Data =====
df_raw = pd.read_csv("dataset/household_power_consumption.csv")

numeric_cols = [
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1',
    'Sub_metering_2', 'Sub_metering_3'
]

# ===== Data Cleaning =====
df = df_raw.copy()
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df = df.drop_duplicates()

# Remove outliers (IQR)
Q1 = df['Global_active_power'].quantile(0.25)
Q3 = df['Global_active_power'].quantile(0.75)
IQR = Q3 - Q1
df = df[
    (df['Global_active_power'] >= Q1 - 1.5 * IQR) &
    (df['Global_active_power'] <= Q3 + 1.5 * IQR)
]

# ===== Regression Data =====
X = df[['Global_intensity']].values
y = df['Global_active_power'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== Models =====
models = {
    'Linear': LinearRegression(),
    'Polynomial2': LinearRegression(),
    'Polynomial3': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

poly_degrees = {
    'Polynomial2': 2,
    'Polynomial3': 3
}

results = {}

# ===== Train & Evaluate =====
for name, model in models.items():
    X_tr, X_te = X_train, X_test

    if name in poly_degrees:
        poly = PolynomialFeatures(poly_degrees[name])
        X_tr = poly.fit_transform(X_train)
        X_te = poly.transform(X_test)

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = (r2, mse)

# ===== Results =====
print("===== Regression Model Comparison =====")
for name, (r2, mse) in results.items():
    print(f"{name}: R²={r2:.4f}, MSE={mse:.4f}")

# ===== Plot =====
sorted_idx = X_test[:, 0].argsort()
X_sorted = X_test[sorted_idx]
y_sorted = y_test[sorted_idx]

plt.figure(figsize=(12, 8))
plt.scatter(X_test, y_test, alpha=0.25, label='Actual Data')

for name, model in models.items():
    X_plot = X_sorted

    if name in poly_degrees:
        poly = PolynomialFeatures(poly_degrees[name])
        X_plot = poly.fit_transform(X_plot)

    y_plot = model.predict(X_plot)
    plt.plot(X_sorted, y_plot, linewidth=2, label=name)

plt.xlabel("Global Intensity (A)")
plt.ylabel("Global Active Power (kW)")
plt.title("Regression Models Comparison")
plt.legend()
plt.grid(True)
plt.show()
