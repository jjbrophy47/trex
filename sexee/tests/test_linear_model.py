"""
Tests the SVM and Kernel logistic regression models.
"""
from sexee.models import linear_model

random_state = 69


def test_svm():

    try:
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
    except Exception:
        print('skipping test_svm!')
        return

    # setup
    data = load_iris()
    X, y = data['data'], data['target']
    model = linear_model.SVM(random_state=random_state).fit(X, y)

    # execute
    res = accuracy_score(y, model.predict(X))

    # assert
    res == 0.96


def test_lr(random_state=69, n_estimators=100):

    try:
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
    except Exception:
        print('skipping test_lr!')
        return

    # setup
    data = load_iris()
    X, y = data['data'], data['target']
    model = linear_model.KernelLogisticRegression().fit(X, y)

    # execute
    res = accuracy_score(y, model.predict(X))

    # assert
    res == 0.96
