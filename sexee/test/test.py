"""
Tests feature path encodings by parsing the raw form of the tree ensembles.
"""
import os
import sys
import json
sys.path.insert(0, os.getcwd() + '/../../')  # for sexee

import lightgbm
import catboost
import xgboost
import numpy as np
from sklearn.datasets import load_iris

from sexee import tree_model


def test_lgb_feature_path_encoding(random_state=69, n_estimators=100):

    # setup
    data = load_iris()
    X, y = data['data'], data['target']
    lgb = lightgbm.LGBMClassifier(random_state=random_state, n_estimators=n_estimators).fit(X, y)
    lgb_model = tree_model.LGBModel(lgb._Booster.dump_model())

    # execute
    encoding = lgb_model.decision_path(X)

    # assert
    assert encoding.shape[0] == len(X)
    assert np.all(encoding[0][:5] == np.array([1, 1, 0, 1, 0]))
    assert np.all(encoding[0][5:14] == np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]))
    assert np.all(encoding[-1][5:14] == np.array([1, 0, 1, 0, 0, 0, 1, 1, 0]))
    print('test_lgb_feature_path_encoding: pass')


def test_xgb_feature_path_encoding(random_state=69, n_estimators=100):

    data = load_iris()
    X, y = data['data'], data['target']
    xgb = xgboost.XGBClassifier(random_state=random_state, n_estimators=n_estimators).fit(X, y)
    xgb_model = tree_model.XGBModel(xgb._Booster.get_dump())

    encoding = xgb_model.decision_path(X)

    assert encoding.shape[0] == len(X)
    assert np.all(encoding[0][:3] == np.array([1, 1, 0]))
    assert np.all(encoding[0][3:12] == np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]))
    assert np.all(encoding[-1][3:12] == np.array([1, 0, 1, 0, 0, 0, 0, 0, 1]))
    print('test_xgb_feature_path_encoding: pass')


def test_cb_feature_path_encoding(random_state=69, n_estimators=100):

    data = load_iris()
    X, y = data['data'], data['target']
    cb = catboost.CatBoostClassifier(random_state=random_state, verbose=0, n_estimators=n_estimators).fit(X, y)
    cb.save_model('.model.json', format='json')
    cb_json = json.load(open('.model.json', 'r'))
    cb_model = tree_model.CBModel(cb_json)

    encoding = cb_model.decision_path(X)

    assert encoding.shape[0] == len(X)
    answer1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0])
    answer2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 1, 0, 1]])
    answer3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0]])
    assert np.all(encoding[0][:127] == answer1)
    assert np.all(encoding[0][127:254] == answer2)
    assert np.all(encoding[-1][127:254] == answer3)
    print('test_cb_feature_path_encoding: pass')

    os.system('rm .model.json')


def main():

    test_lgb_feature_path_encoding()
    test_xgb_feature_path_encoding()
    test_cb_feature_path_encoding()


if __name__ == '__main__':
    main()
