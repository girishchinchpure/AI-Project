# Credit Card Fraud Detection Project

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE


#  Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!\n")
print(df.head())


#  Split data

X = df.drop("Class", axis=1)
y = df["Class"]


#  Handle imbalanced data

# Using SMOTE to balance fraud and non-fraud cases
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(y_resampled.value_counts())


#  Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


#  Train model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Predictions

pred = model.predict(X_test)


#  Evaluation

print("\nModel Accuracy:", accuracy_score(y_test, pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

print("\nClassification Report:\n", classification_report(y_test, pred))