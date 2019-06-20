"""
Experiment: Test to see if we can choose selective points from the training data that represesent the entire
training data.
"""
import argparse

import tqdm
import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

from util import model_util, data_util


def compression(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=20,
                random_state=69, timeit=False, test_ndx=0):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    # get impact from incorrectly predicted test instances
    exp = sexee.TreeExplainer(tree, X_train, y_train)
    svm = exp.get_svm()

    # filter out train instances
    sv_ndx = svm.support_
    print(sv_ndx, len(sv_ndx))
    new_X_train = X_train[sv_ndx]
    new_y_train = y_train[sv_ndx]

    # train new model on cleaned data
    tree = clone(clf).fit(new_X_train, new_y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    parser.add_argument('--test_ndx', metavar='NUM', type=int, default=0, help='Test instance to explain.')
    args = parser.parse_args()
    print(args)
    compression(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.timeit, args.test_ndx)
