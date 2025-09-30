#Homework 2, Problem 3a - Adding penalty to the loss using code from Problem 2a
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Housing.csv')
data = data.drop(columns=['price_per_sqft', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus', 'prefarea'], errors = 'ignore') #drop non-numeric and non-relevant columns

x = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']].values 
y = data['price'].values 

#Using the Standardization section from Problem 2a for the penalty 
scaler_stan = StandardScaler()
x_stan = scaler_stan.fit_transform(x)

m = len(y) #number of instances

x_stan_b = np.c_[np.ones((len(x_stan), 1)), x_stan] #add x0 = 1 to each instance

x_train_stan, x_test_stan, y_train_stan, y_test_stan = train_test_split(x_stan_b, y, test_size=0.2, random_state=42) #80% training, 20% testing

def gradient_descent(x_train, x_test, y_train, y_test, alpha = 0.05, iterations = 1000, lam=0.1):
    m_train = len(y_train)
    m_valid = len(y_test)
    theta = np.zeros(x_train.shape[1])
    train_loss = []
    valid_loss = []

    for i in range(iterations):
        # Training loss
        y_pred_train = x_train.dot(theta)
        error_train = y_pred_train - y_train
        cost_train = (1/(2*m_train)) * np.sum(error_train**2) + (lam/(2*m_train)) * np.sum(theta[1:]**2) 
        train_loss.append(cost_train)

        # Validation loss
        y_pred_valid = x_test.dot(theta)
        error_valid = y_pred_valid - y_test
        cost_valid = (1/(2*m_valid)) * np.sum(error_valid**2)
        valid_loss.append(cost_valid)

        gradients = (1/m_train) * x_train.T.dot(error_train) 
        gradients[1:] += (lam/m_train) * theta[1:]
        theta = theta - alpha * gradients 

    return theta, train_loss, valid_loss

lam_values = [0.01, 0.1, 1]
plt.figure(figsize = (10, 8))

for lam in lam_values:
    theta, train_loss, valid_loss = gradient_descent(x_train_stan, x_test_stan, y_train_stan, y_test_stan, alpha=0.05, iterations=1000, lam=lam)
    plt.plot(train_loss, label=f'Training Loss (lambda={lam})')
    plt.plot(valid_loss, label=f'Validation Loss (lambda={lam})')
    print(f"Lambda: {lam}, Final Parameters - Theta: {theta}")

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Regularization on Standardized Features')
plt.legend()
plt.grid(True)
plt.show()