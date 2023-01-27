"""
Ordinary Least Squares (OLS) regression is a method for estimating the parameters of a linear regression model.
It finds the parameters that minimize the sum of the squared differences between the predicted values and the actual values. 
OLS is a widely used method for fitting linear models and is often used as a benchmark for other methods. 
It assumes that the errors in the model are normally distributed and have constant variance. 
OLS also assumes that there is no multicollinearity among independent variables and that the observations are independent of each other.
"""

import numpy as np
import matplotlib.pyplot as plt


import numpy as np

# Function to calculate the coefficients (b0 and b1)
def OLS(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    mul_xy = np.multiply(x, y)
    sum_xy = np.sum(mul_xy)
    square_xx = np.square(x)
    sum_x2 = np.sum(square_xx)

    # Calculate the coefficients
    b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b0 = (sum_y - b1 * sum_x) / n

    return b0,b1

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

b0,b1 = OLS(x, y)

print("b0 =",b0)
print("b1 =",b1)

# Plot the data points
plt.scatter(x, y)


# Plot the regression line
plt.plot(x, [b0 + b1*i for i in x], 'r')

plt.xlabel('x')
plt.ylabel('y')
plt.title('OLS Linear Regression')

plt.show()