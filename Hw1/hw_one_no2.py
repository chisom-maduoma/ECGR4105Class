import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D3.csv')
x = data.iloc[:, 0:3].values
y = data.iloc[:, 3].values
m = len(y)

x_b = np.c_[np.ones((m, 1)), x]

def gradient_descent(x, y, alpha=0.02, iterations=1000):
    m, n = x.shape
    theta = np.zeros(n)
    loss = []

    for _ in range(iterations):
        y_pred = x.dot(theta)
        error = y_pred - y
        cost = (1/(2*m)) * np.sum(error**2)
        loss.append(cost)

        gradients = (1/m) * x.T.dot(error)
        theta -= alpha * gradients

    return theta, loss

alphas = [0.1, 0.05, 0.01]
results = {}

for alpha in alphas:
    theta, loss = gradient_descent(x_b, y, alpha=alpha, iterations = 1000)
    results[alpha] = (theta, loss)

best_alpha = min(results, key = lambda a: results[a][1][-1])
best_theta = results[best_alpha][0]
best_loss = results[best_alpha][1]

print(f"Best model: Y = {best_theta[0]:0.4f} + {best_theta[1]:0.4f}*x1 + {best_theta[2]:0.4f}*x2 + {best_theta[3]:0.4f}*x3")
print(f"Learning rate: {best_alpha}, Final Loss: {best_loss[-1]:0.4f}")

plt.figure(figsize = (10, 5))
for alpha, (theta, loss) in results.items():
    plt.plot(loss, label = f"alpha = {alpha}")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations (Multivariable Regression)")
plt.legend()
plt.show()

x_new = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
x_new_b = np.c_[np.ones((x_new.shape[0], 1)), x_new]
y_pred = x_new_b.dot(best_theta)

print("Predictions for new data points:")
for x, pred in zip(x_new, y_pred):
    print(f"X={tuple(x)} => Y = {pred:0.4f}")