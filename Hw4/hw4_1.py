#Homework 4, Problem 1 - SVM Classifier for Different Tumor Types on Breast Cancer Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#SVM with different kernels
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
result = []

for k in kernel_types:
    svm_model = SVC(kernel=k, C=1E6, random_state=50)
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    result.append({
        'Kernel': k,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    })

result_df = pd.DataFrame(result)
print(result_df)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

#Plot of metrics for different kernels
plt.figure(figsize=(10,6))
result_melted = result_df.melt(id_vars='Kernel', var_name='Metric', value_name='Score')
sns.barplot(data=result_melted, x='Kernel', y='Score', hue='Metric', palette='viridis')
plt.title('SVM Classifier Performance Metrics Across Different Kernels on Breast Cancer Dataset')
plt.ylim(0, 1.5)
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Compare with Logistic Regression results from Homework 3
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(random_state=0, max_iter=500, warm_start=True)
log_model.fit(x_train, y_train)
y_pred_log = log_model.predict(x_test)

log_acc = metrics.accuracy_score(y_test, y_pred_log)
log_prec = metrics.precision_score(y_test, y_pred_log)
log_rec = metrics.recall_score(y_test, y_pred_log)

print("\n--- Logistic Regression (from Homework 3) ---")
print(f"Accuracy: {log_acc:.4f}")
print(f"Precision: {log_prec:.4f}")
print(f"Recall: {log_rec:.4f}")

# Adding to DataFrame for comparison
compare_df = result_df.copy()
compare_df.loc[len(compare_df)] = {
    'Kernel': 'Logistic Regression',
    'Accuracy': log_acc,
    'Precision': log_prec,
    'Recall': log_rec
}

plt.figure(figsize=(10,6))
compare_melted = compare_df.melt(id_vars='Kernel', var_name='Metric', value_name='Score')
sns.barplot(data=compare_melted, x='Kernel', y='Score', hue='Metric', palette='viridis')
plt.title('Comparison: SVM Kernels vs Logistic Regression')
plt.ylim(0, 1.1)
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)


#2D Visualization using PCA for the best SVM model (RBF kernel)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train_pca = np.array(pca.fit_transform(x_train), dtype=float)
x_test_pca = np.array(pca.transform(x_test), dtype=float)

svm_rbf = SVC(kernel='rbf', C=1E6, random_state=50)
svm_rbf.fit(x_train_pca, y_train)
y_pred_pca = svm_rbf.predict(x_test_pca)

label_map = {0: 'Malignant', 1: 'Benign'}

plt.figure(figsize=(8, 10))
for label in np.unique(y_pred_pca):
    plt.scatter(
        x_test_pca[y_pred_pca == label, 0],
        x_test_pca[y_pred_pca == label, 1],
        label = label_map[label], cmap='autumn', edgecolors='k')
plt.title('SVM Classifier (RBF Kernel)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Predicted Class', labels=['Malignant', 'Benign'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()