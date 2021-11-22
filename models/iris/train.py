import json
import pickle

import sklearn.metrics as metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

with open('iris.pickle', 'wb') as f:
    pickle.dump(model, f)

initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open('iris.onnx', 'wb') as f:
    f.write(onx.SerializeToString())
