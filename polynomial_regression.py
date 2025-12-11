import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Load Cleaned Data =====
df = pd.read_csv("cleaned_smart_home_data.csv")

# ===== Features & Target =====
# هنعمل Predict للـ Global_active_power
features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
target = 'Global_active_power'

X = df[features]
y = df[target]

# ===== Train/Test Split 70/30 =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Linear Regression =====
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict
y_pred = linear_model.predict(X_test)

# ===== Evaluation =====
print("\n===== Linear Regression =====")
print(f"R² Score  : {r2_score(y_test, y_pred):.4f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.3f} kW")
print(f"MAPE      : {mean_absolute_percentage_error(y_test, y_pred)*100:.2f} %")

# Example: Predict on a new sample
new_sample = pd.DataFrame({
    'Global_reactive_power': [0.1],
    'Voltage': [241.8],
    'Global_intensity': [10.5],
    'Sub_metering_1': [0],
    'Sub_metering_2': [1],
    'Sub_metering_3': [0]
})

predicted = linear_model.predict(new_sample)
print(f"Predicted Global Active Power (kW): {predicted[0]:.3f}")

# ===== Visualization =====

# 1. Scatter plot: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Global Active Power (kW)")
plt.ylabel("Predicted Global Active Power (kW)")
plt.title("Actual vs Predicted Global Active Power")
plt.show()

# 2. Histogram of Global Active Power
plt.figure(figsize=(8,5))
sns.histplot(df['Global_active_power'], bins=50, kde=True)
plt.title("Distribution of Global Active Power")
plt.xlabel("Global Active Power (kW)")
plt.ylabel("Frequency")
plt.show()

# 3. Boxplot for Sub_metering_1 as example
plt.figure(figsize=(8,5))
sns.boxplot(y='Sub_metering_1', data=df)
plt.title("Boxplot of Sub_metering_1")
plt.show()
