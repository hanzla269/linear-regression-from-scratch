import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from linear_regression import LinearRegression, StandardScaler


# Load California Housing dataset
data = fetch_california_housing(as_frame=True)

X = data.data.values
y = data.target.values


# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=43
)


# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train model
model = LinearRegression(
    epochs=10000,
    learning_rate=0.01
)

model.fit(X_train_scaled, y_train)


# Make predictions
prediction = model.predict(X_test_scaled)


# Model evaluation
print("First 10 Predictions:")
print(prediction[:10])

print("\nMSE:", model.mse(X_test_scaled, y_test))
print("MAE:", model.mae(X_test_scaled, y_test))
print("R² Score:", model.r2(X_test_scaled, y_test))


# Actual vs Predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    prediction,
    alpha=0.3,
    color="blue"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("California Housing - Actual vs Predicted")

plt.show()
