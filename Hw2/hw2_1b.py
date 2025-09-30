#Homework 2, Problem 1b - Develop Gradient Descent Algorithm to predict house prices based on Area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('Housing.csv')
data = data.drop(columns=['price_per_sqft', 'furnishingstatus'], errors = 'ignore') #drop non-numeric and non-relevant columns

for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    data[col] = data[col].map({'yes': 1, 'no': 0})

x = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']].values 
y = data['price'].values #price
m = len(y) #number of instances

x_b = np.c_[np.ones((m, 1)), x] #add x0 = 1 to each instance

x_train, x_test, y_train, y_test = train_test_split(x_b, y, test_size=0.2, random_state=42) #80% training, 20% testing

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
plt.figure(figsize = (10, 6))

for alpha in learning_rates:
    theta, train_loss, valid_loss = gradient_descent(x_train, x_test, y_train, y_test, alpha=alpha, iterations=1000)
    plt.plot(train_loss, label=f'Training Loss (alpha={alpha})')
    plt.plot(valid_loss, label=f'Validation Loss (alpha={alpha})')
    print(f"Learning Rate: {alpha}, Final Parameters - Theta: {theta}")

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()

print(data.head(15))