import pandas as pd
import numpy as np

print("Loading data...")

# 1. Load data
df = pd.read_csv(r"archive\smart_home_energy_consumption_large.csv")

print("=== Info before cleaning ===")
print("Number of rows and columns:", df.shape)
print("Number of duplicates:", df.duplicated().sum())
print("Number of missing values:\n", df.isnull().sum())

# 2. Cleaning
print("\nCleaning data...")

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric values with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove outliers from 'Energy Consumption (kWh)' only
Q1 = df['Energy Consumption (kWh)'].quantile(0.25)
Q3 = df['Energy Consumption (kWh)'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

before = len(df)
df = df[(df['Energy Consumption (kWh)'] >= lower) & (df['Energy Consumption (kWh)'] <= upper)]
after = len(df)
print(f"Removed {before - after} outlier rows")

# Convert date and time
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time

# Drop unnecessary column (optional)
if 'Home ID' in df.columns:
    df = df.drop('Home ID', axis=1)

# 3. Save cleaned version
df.to_csv("cleaned_smart_home_data.csv", index=False)

print("\nDone!")
print(f"Final number of rows: {len(df):,}")
print("File saved as: cleaned_smart_home_data.csv in the same folder")
