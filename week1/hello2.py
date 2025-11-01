
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, ConfusionMatrixDisplay


df = pd.read_csv('electric_vehicle_analytics.csv')

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())


df.fillna(df.mean(numeric_only=True), inplace=True)


df.dropna(thresh=0.8 * len(df), axis=1, inplace=True)


if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

if 'manufacturer' in df.columns:
    df['manufacturer'] = df['manufacturer'].astype('category')

print("\n--- Cleaned Dataset Info ---")
print(df.info())


plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8))
plt.suptitle('Distribution of Numeric Features')
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


print("\n=== REGRESSION MODEL: Predicting EV Cost ===")


if 'cost_usd' in df.columns:
    features_reg = ['battery_capacity_kWh', 'range_km', 'charging_time_hr']

    features_reg = [col for col in features_reg if col in df.columns]

    X = df[features_reg]
    y = df['cost_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred, color='blue')
    plt.xlabel('Actual Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Actual vs Predicted EV Cost')
    plt.show()
else:
    print("⚠️ 'cost_usd' column not found — skipping regression step.")


print("\n=== CLASSIFICATION MODEL: Predicting Battery Condition ===")


if 'battery_condition' in df.columns:
    features_cls = ['charging_cycles', 'battery_capacity_kWh', 'range_km']
    features_cls = [col for col in features_cls if col in df.columns]

    X = df[features_cls]
    y = df['battery_condition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.title('Confusion Matrix - Battery Condition Classification')
    plt.show()
else:
    print("⚠️ 'battery_condition' column not found — skipping classification step.")

print("\n✅ Machine Learning pipeline completed successfully!")
