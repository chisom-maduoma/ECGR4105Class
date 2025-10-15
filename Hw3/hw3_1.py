#Homework 3, Problem 1 - Logistic Regression Binary Classifier for Positive Diabetes 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, log_loss
import seaborn as sns

diabetes = pd.read_csv('diabetes.csv')

x = diabetes.iloc[:, 0:7].values #Considering all columns except Outcome
y = diabetes.iloc[:, 8].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) #80% train, 20% testing

stan_x = StandardScaler()
x_train = stan_x.fit_transform(x_train)
x_test = stan_x.transform(x_test)

classifier = LogisticRegression(random_state=0, max_iter=500, warm_start=True)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)

print(diabetes.head())
print(diabetes.shape)
print(x_train.shape)
print("Predicted values: ", y_pred)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Confusion Matrix: \n", cm)

class_names=[1,0] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

#Confusion Matrix Plot 
sns.heatmap(pd.DataFrame(cm), annot=True, cmap=plt.cm.Reds, fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Training Loss and Accuracy Plots
losses = []
accuracies = []
for i in range(1, 501):
    classifier.max_iter = i
    classifier.fit(x_train, y_train)
    y_train_pred = classifier.predict(x_train)
    accuracies.append(metrics.accuracy_score(y_train, y_train_pred))
    y_train_prob = classifier.predict_proba(x_train)
    losses.append(log_loss(y_train, y_train_prob))

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