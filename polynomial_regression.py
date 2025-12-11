import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge  # Better than LinearRegression with Polynomial
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Load the data
df = pd.read_csv(r"cleaned_smart_home_data.csv", nrows=100000)

# Columns
cat_cols = ['Appliance Type', 'Season']
num_cols = ['Outdoor Temperature (°C)', 'Household Size']  # Must match exactly the names in the data

# Features and target
X = df[cat_cols + num_cols]  # Fixed: correct concatenation
y = df['Energy Consumption (kWh)']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor + Polynomial + Ridge in one pipeline
poly_model = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', 'passthrough', num_cols),  # Fixed: 'passthrough' not 'pass'
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)  # Fixed: cat_cols
    ])),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Degree 2 to avoid RAM issues
    ('ridge', Ridge(alpha=1.0))  # Ridge to prevent severe overfitting
])

# Train
poly_model.fit(X_train, y_train)  # Fixed: y_train not y

# Predict
y_pred = poly_model.predict(X_test)

# All metrics as requested
print("\n" + "=" * 60)
print("Final Results for Polynomial Regression (Degree 2 + Ridge)")
print("=" * 60)
print(f"R² Score                  : {r2_score(y_test, y_pred):.4f}")
print(f"MAE (Mean Absolute Error) : {mean_absolute_error(y_test, y_pred):.3f} kWh")
print(f"RMSE                      : {np.sqrt(mean_squared_error(y_test, y_pred)):.3f} kWh")
print(f"MAPE (Percentage Error)   : {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f} %")
print(f"Average Real Consumption  : {y.mean():.3f} kWh")
print("=" * 60)

# بيانات جديدة
new_data = pd.DataFrame({
    'Appliance Type': ['Oven'],    # categorical
    'Season': ['Summer'],           # categorical
    'Outdoor Temperature (°C)': [30],
    'Household Size': [5]
})

# Predict مباشرة
predicted_energy = poly_model.predict(new_data)
print(f"Predicted Energy Consumption: {predicted_energy[0]:.3f} kWh")
