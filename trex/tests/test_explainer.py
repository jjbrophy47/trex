"""
Tests the TreeExplainer under different conditions with different tree ensembles.
These tests by no means cover every branch condition.
"""
import numpy as np

import sexee

random_state = 69


def test_gbm_accuracy():

    try:
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import GradientBoostingClassifier
    except Exception:
        print('skipping test_gbm_accuracy!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = GradientBoostingClassifier(random_state=random_state).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='svm', kernel='linear', encoding='leaf_path')

    # execute
    res = accuracy_score(y, m.predict(X))

    # assert
    assert res == 1.0


def test_explainer_predict_proba():

    try:
        from sklearn.datasets import load_iris
        import lightgbm
    except Exception:
        print('skipping test_explainer_predict_proba!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = lightgbm.LGBMClassifier(random_state=random_state).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='lr', kernel='linear', encoding='leaf_output')

    # execute
    res = m.predict_proba(X)

    # assert
    assert res.shape == (150, 3)
    assert np.allclose(np.sum(res, axis=1), np.ones(150))


def test_explainer_decision_function():

    try:
        from sklearn.datasets import load_iris
        import lightgbm
    except Exception:
        print('skipping test_explainer_decision_function!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = lightgbm.LGBMClassifier(random_state=random_state).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='svm', kernel='sigmoid', encoding='leaf_output')

    # execute
    res = m.decision_function(X)

    # assert
    assert res.shape == (150, 3)
    assert np.all(np.argmax(res, axis=1) == m.predict(X))


def test_lgb_similarity():

    try:
        from sklearn.datasets import load_iris
        import lightgbm
    except Exception:
        print('skipping test_lgb_similarity!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = lightgbm.LGBMClassifier(random_state=random_state).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='svm', kernel='linear', encoding='leaf_path')

    # execute
    res1 = m.similarity(X[0].reshape(1, -1))
    res2 = m.similarity(X[-2:])

    # assert
    assert res1[0][0] == 300
    assert res1.shape == (1, 150)
    assert np.all(res1 >= 0)
    assert res2[0][-2] == 300
    assert res2[1][-1] == 300
    assert res2.shape == (2, 150)
    assert np.all(res2 >= 0)


def test_xgb_weight():

    try:
        from sklearn.datasets import load_iris
        import scipy
        import xgboost
    except Exception:
        print('skipping test_xgb_weight!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = xgboost.XGBClassifier(random_state=random_state).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='svm', kernel='poly', encoding='leaf_path')

    # execute
    res = m.get_weight()

    # assert
    assert type(res) == scipy.sparse.csr.csr_matrix
    assert res.shape == (3, 150)
    assert not np.all(res.toarray() >= 0)


def test_cb_explain():

    try:
        from sklearn.datasets import load_iris
        import scipy
        import catboost
    except Exception:
        print('skipping test_cb_explain!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = catboost.CatBoostClassifier(random_state=random_state, verbose=False, n_estimators=10).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='svm', kernel='linear', encoding='leaf_path')

    # execute
    res = m.explain(X[0].reshape(1, -1))
    res2 = m.explain(X[:2])

    # assert
    assert type(res) == scipy.sparse.csr.csr_matrix
    assert res.shape == (1, 150)
    assert not np.all(res.toarray() >= 0)
    assert type(res2) == scipy.sparse.csr.csr_matrix
    assert res2.shape == (2, 150)
    assert not np.all(res2.toarray() >= 0)


def test_rf_binary_classification():

    try:
        import scipy
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        print('skipping test_rf_binary_classification!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    ndx = np.where(y != 2)[0]
    X = X[ndx]
    y = y[ndx]
    tree = RandomForestClassifier(random_state=random_state).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='svm', kernel='rbf', encoding='leaf_path')

    # execute
    res = m.explain(X[0].reshape(1, -1))
    res2 = m.explain(X[:2])

    # assert
    assert type(res) == scipy.sparse.csr.csr_matrix
    assert res.shape == (1, 100)
    assert not np.all(res.toarray() >= 0)
    assert type(res2) == scipy.sparse.csr.csr_matrix
    assert res2.shape == (2, 100)
    assert not np.all(res2.toarray() >= 0)


def test_lgb_logisticregression():

    try:
        from sklearn.datasets import load_iris
        import lightgbm
    except Exception:
        print('skipping test_lgb_logisticregression!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = lightgbm.LGBMClassifier(random_state=random_state).fit(X, y)
    m = sexee.TreeExplainer(tree, X, y, linear_model='lr', kernel='linear', encoding='leaf_output')

    # execute
    res = m.explain(X[0].reshape(1, -1))
    res2 = m.explain(X[:2])

    # assert
    assert type(res) == np.ndarray
    assert res.shape == (1, 150)
    assert not np.all(res >= 0)
    assert type(res2) == np.ndarray
    assert res2.shape == (2, 150)
    assert not np.all(res2 >= 0)


def test_lgb_lr_nonlinear_kernel():

    try:
        from sklearn.datasets import load_iris
        import lightgbm
    except Exception:
        print('skipping test_lgb_lr_nonlinear_kernel!')
        return

    # setup
    data = load_iris()
    X = data['data']
    y = data['target']
    tree = lightgbm.LGBMClassifier(random_state=random_state).fit(X, y)

    # execute
    try:
        sexee.TreeExplainer(tree, X, y, linear_model='lr', kernel='rbf', encoding='leaf_output')
    except Exception:
        assert True
    else:
        assert False
