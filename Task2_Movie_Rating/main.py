# Import Libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Basic Information

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# Data Cleaning

# Remove rows where Rating is missing
df = df.dropna(subset=['Rating'])

# Fill missing categorical values
df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown')
df['Actor 1'] = df['Actor 1'].fillna('Unknown')

# Clean Duration column
df['Duration'] = df['Duration'].str.replace(' min', '', regex=False)
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Clean Votes column
df['Votes'] = df['Votes'].str.replace(',', '', regex=False)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Fill missing numeric values
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# Encoding Categorical Columns

label_encoder = LabelEncoder()

df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Director'] = label_encoder.fit_transform(df['Director'])
df['Actor 1'] = label_encoder.fit_transform(df['Actor 1'])

# Features and Target

X = df[['Genre', 'Director', 'Actor 1', 'Duration', 'Votes']]
y = df['Rating']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model Training

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Feature Importance

importance = model.feature_importances_
feature_names = X.columns

# Create DataFrame for plotting
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Sort values
feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

# Plot
plt.figure(figsize=(8,5))

sns.barplot(
    data=feature_df,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()
