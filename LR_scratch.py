import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load data
data = pd.read_csv('fertilizer_yield.csv')
X = data['Amount of fertilizer (kg/hectare)'].values
y = data['Yield (kg/hectare)'].values

# calculate coefficients
n = len(X)
x_mean = np.mean(X)
y_mean = np.mean(y)
x_std = np.std(X)

b1 = sum((X - x_mean) * (y - y_mean)) / (n * x_std ** 2)
b0 = y_mean - b1 * x_mean

# predict yield for a given amount of fertilizer
fertilizer_amount = 200
predicted_yield = b0 + b1 * fertilizer_amount
print(f'Predicted yield for {fertilizer_amount} kg/hectare of fertilizer: {predicted_yield} kg/hectare')

# calculate performance metrics
y_pred = b0 + b1*X
mse = sum((y - y_pred) ** 2) / n
r2 = 1 - (sum((y - y_pred) ** 2) / sum((y - y_mean) ** 2))

# Print performance metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualization
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.xlabel('Amount of fertilizer (kg/hectare)')
plt.ylabel('Yield (kg/hectare)')
plt.show()


