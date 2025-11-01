import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('electric_vehicle_analytics.csv')

print("Preview of dataset:")
print(df.head())


print("\nMissing values per column:")
print(df.isnull().sum())


df.fillna(df.mean(numeric_only=True), inplace=True)

df.dropna(thresh=0.8 * len(df), axis=1, inplace=True)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

if 'manufacturer' in df.columns:
    df['manufacturer'] = df['manufacturer'].astype('category')


print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())


df.hist(figsize=(12, 8))
plt.suptitle('Distribution of Numeric Features', fontsize=14)
plt.tight_layout()
plt.show()


if 'manufacturer' in df.columns:
    df['manufacturer'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Number of Vehicles by Manufacturer')
    plt.xlabel('Manufacturer')
    plt.ylabel('Count')
    plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


if {'battery_capacity_kWh', 'range_km', 'manufacturer'}.issubset(df.columns):
    sns.scatterplot(data=df, x='battery_capacity_kWh', y='range_km', hue='manufacturer')
    plt.title('Battery Capacity vs Range by Manufacturer')
    plt.show()
