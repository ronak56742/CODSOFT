# Titanic Survival Prediction

# Import libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset

df = pd.read_csv("Titanic-Dataset.csv")

# Basic Information

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# Data Cleaning

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin column because it has too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Convert categorical column Sex into numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked column into dummy variables
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Feature Selection

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]

# Target variable
y = df['Survived']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Creation

model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Model Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Visualizations

# Survival Count Plot
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()