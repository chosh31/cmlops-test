import json
import pickle

import sklearn.metrics as metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

with open('iris.pickle', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='micro')
recall = metrics.recall_score(y_test, y_pred, average='micro')

print({'accuracy': accuracy, 'precision': precision, 'recall': recall})
with open('score.json', 'w') as fd:
    json.dump({'accuracy': accuracy, 'precision': precision, 'recall': recall}, fd, indent=4)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

plt.figure(figsize=(9,9))
sns.heatmap(cnf_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
plt.title(f'Accuracy Score: {accuracy}', size = 15)
plt.tight_layout()
plt.savefig('cnf_matrix.png', dpi=120)
