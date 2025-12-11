import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load first 1000 rows
df = pd.read_csv(r"archive\smart_home_energy_consumption_large.csv", nrows=100000)

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['Appliance Type', 'Season'], drop_first=True)

# Features + Target
X = df_encoded.drop(['Energy Consumption (kWh)', 'Date', 'Time'], axis=1)
y = df_encoded['Energy Consumption (kWh)']

# Polynomial transform (degree=3)
poly = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_train, y_train)

# Predict on test
y_pred = poly_model.predict(X_test)

# Evaluate
print("===== Polynomial Regression (deg=2) =====")
print("R² Score:", r2_score(y_test, y_pred))

# ===== Predict on new data =====
# مثال بيانات جديدة
new_data = pd.DataFrame({
    'Home ID': [101],
    'Outdoor Temperature (°C)': [25],
    'Household Size': [3],
    'Appliance Type_Oven': [1],
    'Appliance Type_Fridge': [0],
    'Appliance Type_Heater': [0],
    'Appliance Type_Dishwasher': [0],
    'Appliance Type_Microwave': [0],
    'Season_Spring': [0],
    'Season_Summer': [0],
    'Season_Winter': [1]
})

# Add missing columns from training if any
train_columns = df_encoded.drop(['Energy Consumption (kWh)','Date','Time'], axis=1).columns
for col in train_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Transform new_data with same polynomial features
new_data_poly = poly.transform(new_data[train_columns])

# Predict
predicted_energy = poly_model.predict(new_data_poly)
print("Predicted Energy Consumption (kWh):", predicted_energy[0])
