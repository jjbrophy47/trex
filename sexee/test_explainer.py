import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

import sexee

data = load_iris()
X = data['data']
y = data['target']

# ndx = np.where(y != 2)[0]
# X = X[ndx]
# y = y[ndx]

tree = GradientBoostingClassifier().fit(X, y)

m = sexee.TreeExplainer(tree, X, y, linear_model='lr', kernel='linear', encoding='leaf_output')
print(m)

print('prediction tests')
print(accuracy_score(y, m.predict(X)))

print('similarity tests')
print(m.similarity(X[0].reshape(1, -1)))
print(m.similarity(X[-2:]))

print('weight tests')
print(m.get_weight())

print('explain tests')
