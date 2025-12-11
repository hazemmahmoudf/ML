import pandas as pd
import numpy as np

print("Loading data...")

# 1. Load data
df = pd.read_csv(r"household_power_consumption.csv")

print("=== Info before cleaning ===")
print("Number of rows and columns:", df.shape)
print("Number of duplicates:", df.duplicated().sum())
print("Number of missing values:\n", df.isnull().sum())

# 2. Cleaning
print("\nCleaning data...")

# Remove duplicates
df = df.drop_duplicates()

# Convert numeric columns from string to float
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # invalid parsing -> NaN

# Fill missing numeric values with mean
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Convert date and time
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time

# Remove outliers from 'Global_active_power' only
Q1 = df['Global_active_power'].quantile(0.25)
Q3 = df['Global_active_power'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

before = len(df)
df = df[(df['Global_active_power'] >= lower) & (df['Global_active_power'] <= upper)]
after = len(df)
print(f"Removed {before - after} outlier rows")

# Drop unnecessary column (optional)
if 'index' in df.columns:
    df = df.drop('index', axis=1)

# 3. Save cleaned version
df.to_csv("cleaned_household_power_data.csv", index=False)

print("\nDone!")
print(f"Final number of rows: {len(df):,}")
print("File saved as: cleaned_household_power_data.csv in the same folder")
