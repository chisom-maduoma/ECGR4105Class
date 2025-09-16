import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D3.csv')

x1 = data.iloc[:, 0].values #Note: iloc-row/column indexing, : - all rows, 0 - first column, values - returns numpy array
x2 = data.iloc[:, 1].values
x3 = data.iloc[:, 2].values
y = data.iloc[:, 3].values

def gradient_descent(x, y, alpha = 0.02, iterations = 1000):
    m = len(y)
    theta0 = 0.0
    theta1 = 0.0
    loss = []

    for _ in range(iterations):
        y_pred = theta0 + (theta1 * x)
        error = y_pred - y
        cost = (1/(2*m)) * np.sum(error**2)
        loss.append(cost)

        dtheta0 = (1/m) * np.sum(error)
        dtheta1 = (1/m) * np.sum(error * x)

        theta0 -= alpha * dtheta0
        theta1 -= alpha * dtheta1

    return theta0, theta1, loss

alphas = [0.1, 0.05, 0.01]
results = {}

for i, xi in enumerate([x1, x2, x3], start = 1 ):
    best_loss = float('inf')
    best_model = None

    for alpha in alphas:
        theta0, theta1, loss = gradient_descent(xi, y, alpha=alpha, iterations = 500)

        if loss[-1] < best_loss:
            best_loss = loss[-1]
            best_model = (theta0, theta1, loss, alpha)
    results[f"x{i}"] = best_model

for var, (theta0, theta1,  loss, alpha) in results.items():
    xi = data.iloc[:, int(var[1]) - 1].values
    plt.figure(figsize = (10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(xi, y, color = 'blue', label = "Data")
    plt.plot(xi, theta0 + (theta1*xi), color = "red", label = f"Model {var}")
    plt.xlabel(var)
    plt.ylabel("Y")
    plt.title(f"Linear Regression Line for ({var}, Y) with alpha = {alpha}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Iterations for {var} with alpha = {alpha}")
    plt.show()

for var, (theta0, theta1, loss, alpha) in results.items():
    print(f"Best model for {var}: Y = {theta0:0.4f} + {theta1:0.4f}*{var}, with alpha = {alpha}, Final Loss = {loss[-1]:0.4f}")

print(data.head())