import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")

# 1. Load data
df = pd.read_csv(r"household_power_consumption.csv")

# ===== Visualization Before Cleaning =====
plt.figure(figsize=(10,5))
sns.histplot(pd.to_numeric(df['Global_active_power'], errors='coerce'), bins=100, kde=True)
plt.title("Global Active Power Distribution - Before Cleaning")
plt.xlabel("Global Active Power (kW)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(y=pd.to_numeric(df['Global_active_power'], errors='coerce'))
plt.title("Global Active Power Boxplot - Before Cleaning")
plt.show()

# ===== Cleaning =====
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

# Remove outliers from 'Global_active_power'
Q1 = df['Global_active_power'].quantile(0.25)
Q3 = df['Global_active_power'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

before = len(df)
df = df[(df['Global_active_power'] >= lower) & (df['Global_active_power'] <= upper)]
after = len(df)
print(f"Removed {before - after} outlier rows")

# Drop unnecessary column
if 'index' in df.columns:
    df = df.drop('index', axis=1)

# ===== Visualization After Cleaning =====
plt.figure(figsize=(10,5))
sns.histplot(df['Global_active_power'], bins=100, kde=True)
plt.title("Global Active Power Distribution - After Cleaning")
plt.xlabel("Global Active Power (kW)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(y=df['Global_active_power'])
plt.title("Global Active Power Boxplot - After Cleaning")
plt.show()

# ===== Save Cleaned Data =====
df.to_csv("cleaned_household_power_data.csv", index=False)
print("\nDone!")
print(f"Final number of rows: {len(df):,}")
print("File saved as: cleaned_household_power_data.csv in the same folder")
