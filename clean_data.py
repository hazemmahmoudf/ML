import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df_raw = pd.read_csv("dataset/household_power_consumption.csv")
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# Info before cleaning
print("=== Info before cleaning ===")
print(f"Number of rows and columns: {df_raw.shape}")
print(f"Number of duplicate rows: {df_raw.duplicated().sum()}")
print("Number of missing values per column:")
print(df_raw.isnull().sum())

# Cleaning
df = df_raw.copy()
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Remove duplicates
before_dup = len(df)
df = df.drop_duplicates()
after_dup = len(df)
duplicates_removed = before_dup - after_dup

# Remove outliers
Q1 = df['Global_active_power'].quantile(0.25)
Q3 = df['Global_active_power'].quantile(0.75)
IQR = Q3 - Q1
before_outlier = len(df)
df = df[(df['Global_active_power'] >= Q1 - 1.5*IQR) & (df['Global_active_power'] <= Q3 + 1.5*IQR)]
after_outlier = len(df)
outliers_removed = before_outlier - after_outlier

# Info after cleaning
print("\n=== Info after cleaning ===")
print(f"Number of rows and columns: {df.shape}")
print(f"Duplicates removed: {duplicates_removed}")
print(f"Outliers removed: {outliers_removed}")

# Show first 10 rows of cleaned data
print("\nFirst 10 rows of cleaned data:")
print(df.head(10))

# Regression Data
X = df['Global_intensity'].values.reshape(-1,1)
y = df['Global_active_power'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin = LinearRegression().fit(X_train, y_train)
y_lin_pred = lin.predict(X_test)

# Polynomial Regression degree 2
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression().fit(X_train_poly, y_train)
y_poly_pred = poly_model.predict(X_test_poly)

# One Figure for all plots (hist + regression + boxplots)
fig, axes = plt.subplots(3, 2, figsize=(14,15))

# 1. Histogram Before cleaning
axes[0,0].hist(pd.to_numeric(df_raw['Global_active_power'], errors='coerce').dropna(), bins=70, color='red', alpha=0.7)
axes[0,0].set_title("Histogram Before Cleaning")
axes[0,0].set_xlabel("Global Active Power (kW)"); axes[0,0].set_ylabel("Frequency")

# 2. Histogram After cleaning
axes[0,1].hist(df['Global_active_power'], bins=70, color='green', alpha=0.7)
axes[0,1].set_title("Histogram After Cleaning")
axes[0,1].set_xlabel("Global Active Power (kW)"); axes[0,1].set_ylabel("Frequency")

# 3. Boxplot Before cleaning
axes[1,0].boxplot(pd.to_numeric(df_raw['Global_active_power'], errors='coerce').dropna(), vert=True)
axes[1,0].set_title("Boxplot Before Cleaning")
axes[1,0].set_ylabel("Global Active Power (kW)")

# 4. Boxplot After cleaning
axes[1,1].boxplot(df['Global_active_power'], vert=True)
axes[1,1].set_title("Boxplot After Cleaning")
axes[1,1].set_ylabel("Global Active Power (kW)")

# 5. Regression Scatter + Linear
axes[2,0].scatter(X_test, y_test, alpha=0.3, label='Data')
axes[2,0].plot(X_test, y_lin_pred, color='red', label='Linear')
axes[2,0].set_title("Linear Regression")
axes[2,0].set_xlabel("Global Intensity (A)"); axes[2,0].set_ylabel("Global Active Power (kW)")
axes[2,0].legend(); axes[2,0].grid(True)

# 6. Regression Scatter + Polynomial
axes[2,1].scatter(X_test, y_test, alpha=0.3, label='Data')
axes[2,1].plot(X_test, y_poly_pred, color='green', label='Polynomial')
axes[2,1].set_title("Polynomial Regression (degree=2)")
axes[2,1].set_xlabel("Global Intensity (A)"); axes[2,1].set_ylabel("Global Active Power (kW)")
axes[2,1].legend(); axes[2,1].grid(True)

plt.tight_layout()
plt.savefig("dataset/cleaning_box_and_regression.png", dpi=300, bbox_inches='tight')
plt.show()
