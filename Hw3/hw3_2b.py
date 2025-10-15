#Homework 3, Problem 2b - Regularization of the Breast Cancer Type Classification (Malignant vs Benign) by adding a weight penalty (L2)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, log_loss
import seaborn as sns

cancer = load_breast_cancer()
cancer_data = cancer.data 
cancer_data.shape 
cancer_input = pd.DataFrame(data=cancer_data)
cancer_input.head()
cancer.target_names

x = cancer.data
y = cancer.target

cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
cancer_df['Malignant vs Benign'] = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50) #80% training & 20% testing

stan_x = StandardScaler()
x_train = stan_x.fit_transform(x_train)
x_test = stan_x.transform(x_test)

classifying = LogisticRegression(random_state=0, max_iter=500, penalty='l2', C=0.1, solver='lbfgs') #L2 Regularization
classifying.fit(x_train, y_train)

y_pred = classifying.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 =metrics.f1_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)

print(f"[0,1] -> {cancer.target_names}") # [0, 1] where 0 = malignant, 1 = benign
print(y_pred)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)
print("Confusion Matrix: \n", cm)

class_names = cancer.target_names
fig, ax = plt.subplots()

#Confusion Matrix Plot
sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True, cmap=plt.cm.Reds, fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label with Penalty')
plt.xlabel('Predicted label with Penalty')

#Training Loss and Accuracy Plots
losses = []
accuracies = []
for i in range(1, 501):
    classifying.max_iter = i
    classifying.fit(x_train, y_train)
    y_train_pred = classifying.predict(x_train)
    accuracies.append(metrics.accuracy_score(y_train, y_train_pred))
    y_train_prob = classifying.predict_proba(x_train)
    losses.append(metrics.log_loss(y_train, y_train_prob))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label="Training Loss", color = 'blue')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss vs. Iterations")

plt.subplot(1, 2, 2)
plt.plot(accuracies, label="Training Accuracy", color = 'red')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs. Iterations")
plt.legend()
plt.show()