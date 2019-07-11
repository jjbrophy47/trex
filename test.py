"""
Simple integration tests to make sure the tree explainer works for all models, encodings, and binary / multi-class
datasets. These do NOT test correctness of the values returned by the tree explainer.
"""
import argparse

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
    impact = explainer.train_impact(X_test[0], similarity=True, weight=True, intercept=True)
    train_ndx, impact_vals, sim, weight, intercept = impact
    assert train_ndx.ndim == 1
    assert impact_vals.ndim == 1
    assert sim.ndim == 1
    assert weight.ndim == 1
    assert isinstance(intercept, float)
    assert len(train_ndx) == len(impact_vals) == len(sim) == len(weight)
    print('test_binaryclass_single_instance_impact_output: pass')


def test_multiclass_single_instance_impact_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='iris')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    impact = explainer.train_impact(X_test[0], similarity=True, weight=True, intercept=True)
    train_ndx, impact_vals, sim, weight, intercept = impact
    assert train_ndx.ndim == 1
    assert impact_vals.ndim == 1
    assert sim.ndim == 1
    assert weight.ndim == 1
    assert isinstance(intercept, float)
    assert len(train_ndx) == len(impact_vals) == len(sim) == len(weight)
    print('test_multiclass_single_instance_impact_output: pass')


def test_binaryclass_multi_instance_impact_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    n_test = 11
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    impact = explainer.train_impact(X_test[:n_test], similarity=True, weight=True, intercept=True)
    train_ndx, impact_vals, sim, weight, intercept = impact
    assert train_ndx.ndim == 1
    assert impact_vals.shape[1] == n_test
    assert sim.shape[1] == n_test
    assert weight.shape[1] == n_test
    assert isinstance(intercept, float)
    print('test_binaryclass_multi_instance_impact_output: pass')

# testing decision_function outputs under different conditions


def test_binaryclass_single_instance_decision_function_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    decision, pred_label = explainer.decision_function(X_test[0], pred_label=True)
    assert len(decision) == 1
    assert len(pred_label) == 1
    print('test_binaryclass_single_instance_decision_function_output: pass')


def test_binaryclass_multi_instance_decision_function_output(n_test=11):
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    decision, pred_label = explainer.decision_function(X_test[:n_test], pred_label=True)
    assert len(decision) == len(pred_label)
    print('test_binaryclass_single_instance_decision_function_output: pass')


def test_multiclass_single_instance_decision_function_output():
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)
    decision, pred_label = explainer.decision_function(X_test[0], pred_label=True)
    assert len(decision) == 1
    assert len(pred_label) == 1
    print('test_multiclass_single_instance_decision_function_output: pass')


def test_multiclass_multi_instance_decision_function_output(n_test=11):

    try:
        X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='breast')
        tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
        explainer = sexee.TreeExplainer(tree, X_train, y_train)
        decision, pred_label = explainer.decision_function(X_test[:n_test], pred_label=True)
    except AssertionError:
        print('test_multiclass_multi_instance_decision_function_output: pass')


def test_binaryclass_multi_instance_impact_equals_iterative_single_instance_impact(n_test=2):
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset='nc17_mfc18')
    tree = lightgbm.LGBMClassifier().fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train)

    impact_1 = explainer.train_impact(X_test[:n_test], similarity=True, weight=True, intercept=True)
    impact_2a = explainer.train_impact(X_test[0], similarity=True, weight=True, intercept=True)
    impact_2b = explainer.train_impact(X_test[1], similarity=True, weight=True, intercept=True)

    ndx1, vals1, sim1, weight1, intercept1 = impact_1
    ndx2a, vals2a, sim2a, weight2a, intercept2a = impact_2a
    ndx2b, vals2b, sim2b, weight2b, intercept2b = impact_2b

    assert np.all(ndx1 == ndx2a)
    assert np.all(ndx1 == ndx2b)
    assert np.allclose(vals1[:, 0], vals2a)
    assert np.allclose(vals1[:, 1], vals2b)
    assert np.allclose(sim1[:, 0], sim2a)
    assert np.allclose(sim1[:, 1], sim2b)
    assert np.allclose(weight1[:, 0], weight2a)
    assert np.allclose(weight1[:, 1], weight2b)
    assert intercept1 == intercept2a == intercept2b
    print('test_binaryclass_multi_instance_impact_equals_iterative_single_instance_impact: pass')


# integration tests

def test_model(model='lgb', encodings=['leaf_path', 'leaf_output'],
               datasets=['iris', 'breast', 'wine', 'nc17_mfc18']):

    for encoding in encodings:
        for dataset in datasets:
            print('\nTEST', model, encoding, dataset)
            example(model=model, encoding=encoding, dataset=dataset, timeit=True)


def main(integration=False):

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

    if integration:
        test_model(model='rf')
        test_model(model='gbm')
        test_model(model='lgb')
        test_model(model='cb')
        test_model(model='xgb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--integration', action='store_true', help='run integration tests.')
    args = parser.parse_args()
    print(args)
    main(integration=args.integration)
