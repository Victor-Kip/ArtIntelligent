import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Nairobi Office Price Ex.csv')
X = data['SIZE'].values
y = data['PRICE'].values

# Normalize the data
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X_normalized = (X - X_mean) / X_std
y_normalized = (y - y_mean) / y_std

# Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * X + c
    dm = (-2 / N) * np.sum(X * (y - y_pred))
    dc = (-2 / N) * np.sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c
# Linear regression training function
def train_linear_regression(X, y, epochs=10, learning_rate=0.001):
    m = np.random.rand()
    c = np.random.rand()
    for epoch in range(epochs):
        y_pred = m * X + c
        error = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch + 1}, Mean Squared Error: {error}")
        m, c = gradient_descent(X, y, m, c, learning_rate)
    return m, c
# normalized data
m_normalized, c_normalized = train_linear_regression(X_normalized, y_normalized, epochs=10, learning_rate=0.001)

m = m_normalized * (y_std / X_std)
c = y_mean + c_normalized * y_std - m * X_mean

#draw regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label=' Best Fit')
plt.xlabel('Size')
plt.ylabel('Price')
plt.legend()
plt.title(' Size vs Price')
plt.show()

# Prediction  100 sq. ft.
predicted_price = m * 47 + c
print(f"Predicted Price : {predicted_price}")
