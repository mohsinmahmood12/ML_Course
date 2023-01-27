"""
Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. 
The goal of linear regression is to find the best-fitting straight line through the data points. 
The line is represented by an equation of the form Y = a + bX, where Y is the dependent variable, X is the independent variable, a is the y-intercept, and b is the slope of the line. 
Linear regression can be used for both simple and multiple regression analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression

# load data
data = pd.read_csv('fertilizer_yield.csv')
X = data['Amount of fertilizer (kg/hectare)'].values.reshape(-1, 1)
y = data['Yield (kg/hectare)'].values

# perform linear regression
reg = LinearRegression().fit(X, y)

# Print coefficients
print(f'Intercept: {reg.intercept_}')
print(f'Slope: {reg.coef_[0]}')

# predict yield for a given amount of fertilizer
fertilizer_amount = 200
predicted_yield = reg.predict([[fertilizer_amount]])
print(f'Predicted yield for {fertilizer_amount} kg/hectare of fertilizer: {predicted_yield} kg/hectare')

# Performance metric
# Mean absolute error
y_pred = reg.predict(X)
mae = mean_absolute_error(y, y_pred)
print(f'Mean absolute error: {mae}')

# Mean squared error
mse = mean_squared_error(y, y_pred)
print(f'Mean squared error: {mse}')

# Root mean squared error
rmse = sqrt(mean_squared_error(y, y_pred))
print(f'Root mean squared error: {rmse}')

# R-squared
r2 = r2_score(y, y_pred)
print(f'R-squared: {r2}')

# Visulization
plt.scatter(X, y, color='blue')
plt.plot(X, reg.predict(X), color='red')
plt.xlabel('Amount of fertilizer (kg/hectare)')
plt.ylabel('Yield (kg/hectare)')
plt.title('Fertilizer yield')
plt.show()


