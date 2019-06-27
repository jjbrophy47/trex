"""
Simple integration tests to make sure the tree explainer works for all models, encodings, and binary / multi-class
datasets. These do NOT test correctness of the values returned by the tree explainer.
"""
import lightgbm
import numpy as np

import sexee
from util import data_util
from experiments.scripts.sample import example


# testing behavior for multiple instances for binary and multiclass

def test_multiclass_one_test_instance():

    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='iris')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    explainer.train_impact(X_test[0])
    print('test_multiclass_one_test_instance: pass')


def test_multiclass_fail_multiple_test_instances():

    try:
        X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='iris')
        tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
        explainer = sexee.TreeExplainer(tree, X_train, y_train)
        explainer.train_impact(X_test[:10])
    except AssertionError:
        print('test_multiclass_fail_multiple_test_instances: pass')


def test_binaryclass_one_test_instance():

    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    explainer.train_impact(X_test[0])
    print('test_binaryclass_one_test_instance: pass')


def test_binaryclass_multiple_test_instances():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    explainer.train_impact(X_test[0])
    print('test_binaryclass_multiple_test_instances: pass')


# testing train impact outputs under different conditions

def test_binaryclass_single_instance_impact_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    impact = explainer.train_impact(X_test[0], similarity=True, weight=True)
    train_ndx, impact_vals, sim, weight = impact
    assert train_ndx.ndim == 1
    assert impact_vals.ndim == 1
    assert sim.ndim == 1
    assert weight.ndim == 1
    assert len(train_ndx) == len(impact_vals) == len(sim) == len(weight)
    print('test_binaryclass_single_instance_impact_output: pass')


def test_multiclass_single_instance_impact_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='iris')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    impact = explainer.train_impact(X_test[0], similarity=True, weight=True)
    train_ndx, impact_vals, sim, weight = impact
    assert train_ndx.ndim == 1
    assert impact_vals.ndim == 1
    assert sim.ndim == 1
    assert weight.ndim == 1
    assert len(train_ndx) == len(impact_vals) == len(sim) == len(weight)
    print('test_multiclass_single_instance_impact_output: pass')


def test_binaryclass_multi_instance_impact_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    n_test = 11
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    impact = explainer.train_impact(X_test[:n_test], similarity=True, weight=True)
    train_ndx, impact_vals, sim, weight = impact
    assert train_ndx.ndim == 1
    assert impact_vals.shape[1] == n_test
    assert sim.shape[1] == n_test
    assert weight.shape[1] == n_test
    print('test_binaryclass_multi_instance_impact_output: pass')

# testing decision_function outputs under different conditions


def test_binaryclass_single_instance_decision_function_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    decision, pred_svm = explainer.decision_function(X_test[0], pred_svm=True)
    assert isinstance(decision, float)
    assert isinstance(pred_svm, int)
    print('test_binaryclass_single_instance_decision_function_output: pass')


def test_binaryclass_multi_instance_decision_function_output(n_test=11):
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    decision, pred_svm = explainer.decision_function(X_test[:n_test], pred_svm=True)
    assert len(decision) == len(pred_svm)
    print('test_binaryclass_single_instance_decision_function_output: pass')


def test_multiclass_single_instance_decision_function_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    decision, pred_svm = explainer.decision_function(X_test[0], pred_svm=True)
    assert isinstance(decision, float)
    assert isinstance(pred_svm, int)
    print('test_multiclass_single_instance_decision_function_output: pass')


def test_multiclass_multi_instance_decision_function_output(n_test=11):

    try:
        X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
        tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
        explainer = sexee.TreeExplainer(tree, X_train, y_train)
        decision, pred_svm = explainer.decision_function(X_test[:n_test], pred_svm=True)
    except AssertionError:
        print('test_multiclass_multi_instance_decision_function_output: pass')


def test_binaryclass_multi_instance_impact_equals_iterative_single_instance_impact(n_test=2):
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='medifor')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)

    impact_1 = explainer.train_impact(X_test[:n_test], similarity=True, weight=True)
    impact_2a = explainer.train_impact(X_test[0], similarity=True, weight=True)
    impact_2b = explainer.train_impact(X_test[1], similarity=True, weight=True)

    ndx1, vals1, sim1, weight1 = impact_1
    ndx2a, vals2a, sim2a, weight2a = impact_2a
    ndx2b, vals2b, sim2b, weight2b = impact_2b

    assert np.all(ndx1 == ndx2a)
    assert np.all(ndx1 == ndx2b)
    assert np.all(vals1[:, 0] == vals2a)
    assert np.all(vals1[:, 1] == vals2b)
    assert np.all(sim1[:, 0] == sim2a)
    assert np.all(sim1[:, 1] == sim2b)
    assert np.all(weight1[:, 0] == weight2a)
    assert np.all(weight1[:, 1] == weight2b)
    print('test_binaryclass_multi_instance_impact_equals_iterative_single_instance_impact: pass')


# integration tests

def test_model(model='lgb', encodings=['tree_path', 'tree_output'],
               datasets=['iris', 'breast', 'wine', 'medifor']):

    for encoding in encodings:
        for dataset in datasets:
            print('\nTEST', model, encoding, dataset)
            example(model=model, encoding=encoding, dataset=dataset, timeit=True)


def main():

    # test execution flow for different conditions
    test_multiclass_one_test_instance()
    test_multiclass_fail_multiple_test_instances()
    test_binaryclass_one_test_instance()
    test_binaryclass_multiple_test_instances()

    # testing explanation outputs for the single-instance case
    test_binaryclass_single_instance_impact_output()
    test_multiclass_single_instance_impact_output()
    test_binaryclass_multi_instance_impact_output()

    # testing decision outputs
    test_binaryclass_single_instance_decision_function_output()
    test_binaryclass_multi_instance_decision_function_output()
    test_multiclass_single_instance_decision_function_output()
    test_multiclass_multi_instance_decision_function_output()

    test_binaryclass_multi_instance_impact_equals_iterative_single_instance_impact()

    test_model(model='rf')
    test_model(model='gbm')
    test_model(model='lgb')
    test_model(model='cb')
    test_model(model='xgb')


if __name__ == '__main__':
    main()
