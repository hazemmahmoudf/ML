import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Load first 1000 rows
df = pd.read_csv(r"cleaned_smart_home_data.csv", nrows=100000)

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['Appliance Type', 'Season'], drop_first=True)

# Features + Target
X = df_encoded.drop(['Energy Consumption (kWh)', 'Date', 'Time'], axis=1)
y = df_encoded['Energy Consumption (kWh)']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict
y_pred = linear_model.predict(X_test)

# Evaluate
print("===== Linear Regression =====")
print(f"R² Score  : {r2_score(y_test, y_pred):.4f}")
print(f"MAE       : {mean_absolute_error(y_test, y_pred):.3f} kWh")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test, y_pred)):.3f} kWh")
print(f"MAPE      : {mean_absolute_percentage_error(y_test, y_pred)*100:.2f} %")

# Example: Predict on new data
model_columns = X_train.columns

new_data = pd.DataFrame({
    'Home ID': [101],
    'Outdoor Temperature (°C)': [25],
    'Household Size': [3],
    'Appliance Type_Oven': [1],
    'Season_Winter': [1]
})

# Add missing columns
for col in model_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Reorder columns like training set
new_data = new_data[model_columns]

# Predict
predicted_energy = linear_model.predict(new_data)
print(f"Predicted Energy Consumption (kWh): {predicted_energy[0]:.3f}")
