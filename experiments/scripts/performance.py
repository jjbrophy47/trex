"""
Experiment: Tests tree ensemble v SVM v SVM trained on tree ensemble feature representations for performance.
If an SVM is already as good as a tree ensemble, there is no need to explain a tree ensemble with an SVM.
"""
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())  # for influence_boosting

import numpy as np
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import sexee
from util import model_util, data_util, exp_util


def performance(model_type='lgb', encoding='tree_path', dataset='iris', n_estimators=100, random_state=69,
                gridsearch=False, verbose=0, data_dir='data'):
    """
    Main method comparing performance of tree ensembles and svm models.
    """

    # get model and data
    clf = model_util.get_classifier(model_type, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    print('train instances: {}'.format(len(X_train)))
    print('num features: {}'.format(X_train.shape[1]))

    # train a tree ensemble
    print('\ntree ensemble')
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test=X_test, y_test=y_test)

    # train an svm
    print('\nsvm')
    clf = SVC(gamma='auto')
    if gridsearch:
        param_grid = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
        gs = GridSearchCV(clf, param_grid, cv=2, verbose=verbose).fit(X_train, y_train)
        svm = gs.best_estimator_
        print(gs.best_params_)
    else:
        svm = SVC(C=0.1, gamma='auto').fit(X_train, y_train)
    model_util.performance(svm, X_train, y_train, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--gridsearch', action='store_true', default=False, help='gridsearch for SVM model.')
    parser.add_argument('--verbose', metavar='LEVEL', default=0, type=int, help='verbosity of gridsearch output.')
    args = parser.parse_args()
    print(args)
    performance(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.gridsearch, args.verbose)
