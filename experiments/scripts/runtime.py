"""
Experiment: Compare runtimes for explaining a single test instance for different methods.
"""
import time
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())  # for influence_boosting

import numpy as np
from sklearn.base import clone

import sexee
from util import model_util, data_util, exp_util


def sexee_method(test_ndx, X_test, model, X_train, y_train, encoding, random_state=69):
    """Explains the predictions of each test instance."""

    start = time.time()
    explainer = sexee.TreeExplainer(model, X_train, y_train, encoding=encoding, random_state=random_state)
    fine_tune = time.time() - start

    start = time.time()
    explainer.train_impact(X_test[test_ndx])
    test_time = time.time() - start

    return fine_tune, test_time


def influence_method(model, test_ndx, X_train, y_train, X_test, y_test, inf_k):
    """
    Computes the influence on each test instance if train instance i were upweighted/removed.
    This uses the fastleafinfluence method by Sharchilev et al.
    """

    start = time.time()
    leaf_influence = exp_util.get_influence_explainer(model, X_train, y_train, inf_k)
    fine_tune = time.time() - start

    start = time.time()
    exp_util.influence_explain_instance(leaf_influence, test_ndx, X_train, X_test, y_test)
    test_time = time.time() - start

    return fine_tune, test_time


def runtime(model_type='lgb', encoding='tree_path', dataset='iris', n_estimators=100, random_state=69,
            inf_k=None, repeats=10):
    """
    Main method that trains a tree ensemble, then compares the runtime of different methods to explain
    a random subset of test instances.
    """

    sexee_fine_tune, sexee_test_time = [], []
    inf_fine_tune, inf_test_time = [], []

    for i in range(repeats):
        print('\nrun {}'.format(i))
        random_state += 10

        # get model and data
        clf = model_util.get_classifier(model_type, n_estimators=n_estimators, random_state=random_state)
        X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state)

        print('train instances: {}'.format(len(X_train)))
        print('num features: {}'.format(X_train.shape[1]))

        # train a tree ensemble
        model = clone(clf).fit(X_train, y_train)
        model_util.performance(model, X_test=X_test, y_test=y_test)

        # randomly pick test instances to explain
        np.random.seed(random_state)
        test_ndx = np.random.choice(len(y_test), size=1, replace=False)

        # sexee method
        print('sexee...')
        fine_tune, test_time = sexee_method(test_ndx, X_test, model, X_train, y_train, encoding, random_state)
        print('fine tune: {:.3f}s'.format(fine_tune))
        print('test time: {:.3f}s'.format(test_time))
        sexee_fine_tune.append(fine_tune)
        sexee_test_time.append(test_time)

        # influence method
        if model_type == 'cb' and inf_k is not None:
            print('leafinfluence...')
            fine_tune, test_time = influence_method(model, test_ndx, X_train, y_train, X_test, y_test, inf_k)
            print('fine tune: {:.3f}s'.format(fine_tune))
            print('test time: {:.3f}s'.format(test_time))
            inf_fine_tune.append(fine_tune)
            inf_test_time.append(test_time)

    sexee_fine_tune = np.array(sexee_fine_tune)
    sexee_test_time = np.array(sexee_test_time)
    print('\nsexee')
    print('fine tuning: {:.3f}s +/- {:.3f}s'.format(sexee_fine_tune.mean(), sexee_fine_tune.std()))
    print('test time: {:.3f}s +/- {:.3f}s'.format(sexee_test_time.mean(), sexee_test_time.std()))

    if model_type == 'cb' and inf_k is not None:
        inf_fine_tune = np.array(inf_fine_tune)
        inf_test_time = np.array(inf_test_time)
        print('\nleafinfluence')
        print('fine tuning: {:.3f}s +/- {:.3f}s'.format(inf_fine_tune.mean(), inf_fine_tune.std()))
        print('test time: {:.3f}s +/- {:.3f}s'.format(inf_test_time.mean(), inf_test_time.std()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--inf_k', default=None, type=int, help='Number of leaves for leafinfluence.')
    parser.add_argument('--repeats', default=10, type=int, help='Number of times to repeat the experiment.')
    args = parser.parse_args()
    print(args)
    runtime(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.inf_k, args.repeats)
