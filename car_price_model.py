import pandas as pd

# Load data
df = pd.read_csv("car_data.csv")

# Print first 5 rows
print(df.head())

# Check columns
print(df.columns)

# Drop the 'Car_Name' column
df.drop(['Car_Name'], axis=1, inplace=True)

# Create new feature 'Car_Age'
df['Car_Age'] = 2020 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

# Replace categorical with dummies
df = pd.get_dummies(df, drop_first=True)

print("Updated columns:", df.columns)

X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

import joblib
joblib.dump(model, "car_price_model.pkl")

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
