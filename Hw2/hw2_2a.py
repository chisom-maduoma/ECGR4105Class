#Homework 2, Problem 2a - Repeating Problem 1a but with normalization and standardization to stop the overfitting
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

#Normalization
scaler_norm = MinMaxScaler()
x_norm = scaler_norm.fit_transform(x)

#Standardization
scaler_stan = StandardScaler()
x_stan = scaler_stan.fit_transform(x)

m = len(y) 

x_norm_b = np.c_[np.ones((len(x_norm), 1)), x_norm] 
x_stan_b = np.c_[np.ones((len(x_stan), 1)), x_stan] 

x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(x_norm_b, y, test_size=0.2, random_state=42) #80% training, 20% testing
x_train_stan, x_test_stan, y_train_stan, y_test_stan = train_test_split(x_stan_b, y, test_size=0.2, random_state=42) #80% training, 20% testing

def gradient_descent(x_train, x_test, y_train, y_test, alpha = 0.05, iterations = 1000):
    m_train = len(y_train)
    m_valid = len(y_test)
    theta = np.zeros(x_train.shape[1])
    train_loss = []
    valid_loss = []

    for i in range(iterations):
        # Training loss
        y_pred_train = x_train.dot(theta)
        error_train = y_pred_train - y_train
        cost_train = (1/(2*m_train)) * np.sum(error_train**2)
        train_loss.append(cost_train)

        # Validation loss
        y_pred_valid = x_test.dot(theta)
        error_valid = y_pred_valid - y_test
        cost_valid = (1/(2*m_valid)) * np.sum(error_valid**2)
        valid_loss.append(cost_valid)

        gradients = (1/m_train) * x_train.T.dot(error_train) 
        theta = theta - alpha * gradients 

    return theta, train_loss, valid_loss

learning_rates = [0.01, 0.05, 0.1]
plt.figure(figsize = (10, 8))

for alpha in learning_rates:
    theta, train_loss, valid_loss = gradient_descent(x_train_norm, x_test_norm, y_train_norm, y_test_norm, alpha=alpha, iterations=1000)
    plt.plot(train_loss, label=f'Normalization Training Loss (alpha={alpha})')
    plt.plot(valid_loss, label=f'Normalization Validation Loss (alpha={alpha})')
    print(f"Normalization Learning Rate: {alpha}, Final Parameters - Theta: {theta}")

for alpha in learning_rates:
    theta, train_loss, valid_loss = gradient_descent(x_train_stan, x_test_stan, y_train_stan, y_test_stan, alpha=alpha, iterations=1000)
    plt.plot(train_loss, label=f'Standardization Training Loss (alpha={alpha})')
    plt.plot(valid_loss, label=f'Standardization Validation Loss (alpha={alpha})')
    print(f"Standardization Learning Rate: {alpha}, Final Parameters - Theta: {theta}")

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()