import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Load Data =====
df = pd.read_csv(r"dataset/cleaned_household_power_data.csv", nrows=100000)

# Target: Predict Global_active_power
target_col = 'Global_active_power'

# Features: استخدام باقي الأعمدة الرقمية
feature_cols = ['Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

X = df[feature_cols]
y = df[target_col]

# ===== Train/Test Split 70/30 =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Linear Regression =====
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict
y_pred = linear_model.predict(X_test)

# Evaluate
print("\n===== Linear Regression =====")
print(f"R² Score  : {r2_score(y_test, y_pred):.4f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.3f} kW")
print(f"MAPE      : {mean_absolute_percentage_error(y_test, y_pred)*100:.2f} %")

# Example: Predict on new data
new_data = pd.DataFrame({
    'Global_reactive_power': [0.1],
    'Voltage': [240],
    'Global_intensity': [10],
    'Sub_metering_1': [0],
    'Sub_metering_2': [1],
    'Sub_metering_3': [2]
})

predicted_energy = linear_model.predict(new_data)
print(f"Predicted Global Active Power (kW): {predicted_energy[0]:.3f}")

# ===== Visualization: Scatter + Boxplots together =====
fig, axes = plt.subplots(2, 1, figsize=(12,12))

# 1️⃣ Scatter: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.3)
axes[0].set_xlabel("Actual Global Active Power (kW)")
axes[0].set_ylabel("Predicted Global Active Power (kW)")
axes[0].set_title("Actual vs Predicted Global Active Power")
axes[0].grid(True)

# 2️⃣ Boxplots for Sub_meterings
sns.boxplot(data=df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']], ax=axes[1])
axes[1].set_title('Distribution of Sub Metering')

plt.tight_layout()
# Save figure
plt.savefig("dataset/linear_scatter_and_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()
