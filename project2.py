# House Price Prediction Project

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("Housing.csv")

print("Dataset preview:\n")
print(df.head())


#Step 2: Data preprocessing

# Fill missing values
df.fillna(df.median(), inplace=True)

# Convert categorical variables into numeric
df = pd.get_dummies(df, drop_first=True)


# Split data

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Feature scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train models

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Random Forest (for comparison)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


#  Predictions

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)


#  Evaluation

print("\n----- Linear Regression -----")
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))


print("\n----- Random Forest -----")
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))


#  Feature importance (Random Forest)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop features affecting house price:\n")
print(importance.head(10))