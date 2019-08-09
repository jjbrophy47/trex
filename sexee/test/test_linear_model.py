import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score, accuracy_score

import linear_model

data = load_iris()
X = data['data']
y = data['target']

# ndx = np.where(y != 2)[0]
# X = X[ndx]
# y = y[ndx]

# m = linear_model.KernelLogisticRegression().fit(X, y)
m = linear_model.SVM().fit(X, y)

# print(roc_auc_score(y, m.predict_proba(X)[:, 1]))
print(accuracy_score(y, m.predict(X)))
