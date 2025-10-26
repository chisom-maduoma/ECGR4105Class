#Homework 4, Problem 2 - SVR Regression Model to Predict Housing Prices
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing_data = pd.read_csv('Housing.csv')

for col in housing_data.select_dtypes(include=['object']).columns:
    housing_data[col] = housing_data[col].map({'yes': 1, 'no': 0})


x = housing_data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                  'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']]

y = housing_data['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# SVR with different kernels
svr_rbf = SVR(kernel='rbf', C=1E6, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1E6)
svr_poly = SVR(kernel='poly', C=1E6, degree=2)

y_rbf = svr_rbf.fit(x_train_scaled, y_train).predict(x_test_scaled)
y_lin = svr_lin.fit(x_train_scaled, y_train).predict(x_test_scaled)
y_poly = svr_poly.fit(x_train_scaled, y_train).predict(x_test_scaled)

lw = 2
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_rbf, color='green', label='RBF model', alpha=0.7)
plt.scatter(y_test, y_lin, color='navy', lw=lw, label='Linear model', alpha=0.7)
plt.scatter(y_test, y_poly, color='purple', lw=lw, label='Polynomial model', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=lw, label='Ideal fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

#Metrics for SVR models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

metrics = {}

for name, y_pred in zip(['RBF', 'Linear', 'Polynomial'], [y_rbf, y_lin, y_poly]):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

# Print the results
for kernel, vals in metrics.items():
    print(f"{kernel} kernel:")
    print(f"  Mean Squared Error: {vals['MSE']:.2f}")
    print(f"  Mean Absolute Error: {vals['MAE']:.2f}")
    print(f"  R2 Score: {vals['R2']:.3f}\n")