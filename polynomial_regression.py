import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Load Cleaned Data =====
df = pd.read_csv("dataset/cleaned_household_power_data.csv")

# ===== Features & Target =====
features = ['Global_reactive_power', 'Voltage', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
target = 'Global_active_power'

X = df[features]
y = df[target]

# ===== Train/Test Split 70/30 =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Polynomial Regression + Ridge =====
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0))
])

# Train
poly_model.fit(X_train, y_train)

# Predict
y_pred = poly_model.predict(X_test)

# ===== Metrics =====
print("\n" + "="*60)
print("Polynomial Regression (Degree 2 + Ridge)")
print("="*60)
print(f"R² Score                  : {r2_score(y_test, y_pred):.4f}")
print(f"MAE (Mean Absolute Error) : {mean_absolute_error(y_test, y_pred):.3f} kW")
print(f"RMSE                      : {np.sqrt(mean_squared_error(y_test, y_pred)):.3f} kW")
print(f"MAPE (Percentage Error)   : {mean_absolute_percentage_error(y_test, y_pred)*100:.2f} %")
print(f"Average Real Global Active Power : {y.mean():.3f} kW")
print("="*60)

# Example: Predict on a new sample
new_sample = pd.DataFrame({
    'Global_reactive_power': [0.1],
    'Voltage': [241.8],
    'Global_intensity': [10.5],
    'Sub_metering_1': [0],
    'Sub_metering_2': [1],
    'Sub_metering_3': [0]
})

predicted = poly_model.predict(new_sample)
print(f"Predicted Global Active Power (kW): {predicted[0]:.3f}")

# ===== Visualization in One Figure =====
fig, axs = plt.subplots(3, 2, figsize=(16,18))

# 1. Scatter plot: Actual vs Predicted
axs[0,0].scatter(y_test, y_pred, alpha=0.3)
axs[0,0].set_xlabel("Actual Global Active Power (kW)")
axs[0,0].set_ylabel("Predicted Global Active Power (kW)")
axs[0,0].set_title("Actual vs Predicted Global Active Power (Polynomial)")

# 2. Histogram of Global Active Power
sns.histplot(df['Global_active_power'], bins=50, kde=True, ax=axs[0,1])
axs[0,1].set_title("Distribution of Global Active Power")
axs[0,1].set_xlabel("Global Active Power (kW)")
axs[0,1].set_ylabel("Frequency")

# 3. Boxplot for Sub_metering_1
sns.boxplot(y='Sub_metering_1', data=df, ax=axs[1,0])
axs[1,0].set_title("Boxplot of Sub_metering_1")

# 4. Boxplot for Sub_metering_2
sns.boxplot(y='Sub_metering_2', data=df, ax=axs[1,1])
axs[1,1].set_title("Boxplot of Sub_metering_2")

# 5. Boxplot for Sub_metering_3
sns.boxplot(y='Sub_metering_3', data=df, ax=axs[2,0])
axs[2,0].set_title("Boxplot of Sub_metering_3")

# 6. Empty plot (optional)
axs[2,1].axis('off')

plt.tight_layout()
# Save figure to dataset folder
plt.savefig("dataset/polynomial_ridge_all_visuals.png", dpi=300, bbox_inches='tight')
plt.show()
