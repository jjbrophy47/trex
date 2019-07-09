"""
Exploration: Take the top k worst missed instances, and find the most impactful train instances.
    Then plot the most impactful train instances with the test instances and see if any insight can be gleaned.
"""
import argparse

import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from util import model_util, data_util, exp_util


def _sort_impact(sv_ndx, impact):
    """Sorts support vectors by absolute impact values."""

    impact = np.sum(impact, axis=1)
    impact_list = zip(sv_ndx, impact)
    impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)

    sv_ndx, impact = zip(*impact_list)
    sv_ndx = np.array(sv_ndx)
    return sv_ndx, impact


def _tsne(X_feature, explainer, encoding='tree_output', pca_components=50, random_state=69, verbose=0):

    if encoding in ['tree_output', 'tree_path']:
        X_feature = explainer.transform(X_feature)

    if X_feature.shape[0] > pca_components and X_feature.shape[1] > pca_components:
        if encoding == 'tree_path':
            X_feature = TruncatedSVD(n_components=pca_components, random_state=random_state).fit_transform(X_feature)
        else:
            X_feature = PCA(n_components=pca_components, random_state=random_state).fit_transform(X_feature)

    X_embed = TSNE(verbose=verbose, random_state=random_state).fit_transform(X_feature)
    return X_embed


def misclassification(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100, random_state=69,
                      plot_loss=False, plot_all_sv=False, test_subset=None, train_subset=None, test_type=None,
                      data_dir='data', pca_components=50, verbose=0, alpha=0.75, show_performance=False):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_feature=True)
    X_train, X_test, y_train, y_test, label, feature = data

    # fit a tree ensemble
    tree = clone(clf).fit(X_train, y_train)

    if show_performance:
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    # get worst missed predictions
    test_dist = exp_util.instance_loss(tree.predict_proba(X_test), y_test)
    test_dist_ndx = np.argsort(test_dist)[::-1]
    test_dist = test_dist[test_dist_ndx]
    X_test_miss = X_test[test_dist_ndx][:test_subset]
    y_test_miss = y_test[test_dist_ndx][:test_subset]

    # plot test instance loss
    if plot_loss:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(test_dist)), test_dist)
        ax.set_ylabel('l1 loss')
        ax.set_xlabel('test instances sorted by loss')
        plt.show()

    # filter only test instances with a specific label
    if test_type == 'manip':
        type_ndx = np.where(y_test_miss == 1)[0]
        X_test_miss = X_test_miss[type_ndx]
        y_test_miss = y_test_miss[type_ndx]
    if test_type == 'non_manip':
        type_ndx = np.where(y_test_miss == 0)[0]
        X_test_miss = X_test_miss[type_ndx]
        y_test_miss = y_test_miss[type_ndx]

    # compute most impactful train instances on the test instances
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding)
    sv_ndx, impact = explainer.train_impact(X_test_miss)
    sv_ndx, impact = _sort_impact(sv_ndx, impact)

    if show_performance:
        print(explainer)

    # get indices for each class
    test0 = np.where(y_test_miss == 0)[0]
    test1 = np.where(y_test_miss == 1)[0]

    if plot_all_sv:

        # get data for support vectors
        X_train_sv = X_train[sv_ndx]
        y_train_sv = y_train[sv_ndx]

        train0 = np.where(y_train_sv == 0)[0]
        train1 = np.where(y_train_sv == 1)[0]

        # compute tsne embedding similarity
        X_all = np.concatenate([X_train_sv, X_test_miss])
        Xe = _tsne(X_all, explainer, encoding, pca_components, random_state, verbose)
        Xe_sv = Xe[:len(X_train_sv)]
        Xe_test = Xe[len(X_train_sv):]

        # plot results
        fig, ax = plt.subplots()

        ax.scatter(Xe_sv[:, 0][train0], Xe_sv[:, 1][train0], color='blue', alpha=alpha, label='train (non-manip)',
                   marker='1')
        ax.scatter(Xe_sv[:, 0][train1], Xe_sv[:, 1][train1], color='red', alpha=alpha, label='train (manip)',
                   marker='+')
        ax.scatter(Xe_sv[:, 0][:train_subset], Xe_sv[:, 1][:train_subset], color='green', alpha=alpha,
                   label='train (most impact)', marker='*')

        ax.scatter(Xe_test[:, 0][test0], Xe_test[:, 1][test0], color='cyan', alpha=alpha, label='test (non-manip)',
                   marker='2')
        ax.scatter(Xe_test[:, 0][test1], Xe_test[:, 1][test1], color='orange', alpha=alpha, label='test (manip)',
                   marker='x')

        ax.set_xlabel('tsne 0')
        ax.set_ylabel('tsne 1')
        ax.legend()
        plt.show()

    else:

        # get data for support vectors
        X_train_sv = X_train[sv_ndx][:train_subset]
        y_train_sv = y_train[sv_ndx][:train_subset]

        train0 = np.where(y_train_sv == 0)[0]
        train1 = np.where(y_train_sv == 1)[0]

        # compute tsne embedding similarity
        X_all = np.concatenate([X_train_sv, X_test_miss])
        Xe = _tsne(X_all, explainer, encoding, pca_components, random_state, verbose)
        Xe_sv = Xe[:len(X_train_sv)]
        Xe_test = Xe[len(X_train_sv):]

        # plot results
        fig, ax = plt.subplots()

        ax.scatter(Xe_sv[:, 0][train0], Xe_sv[:, 1][train0], color='blue', alpha=alpha, label='train (non-manip)',
                   marker='1')
        ax.scatter(Xe_sv[:, 0][train1], Xe_sv[:, 1][train1], color='red', alpha=alpha, label='train (manip)',
                   marker='+')

        ax.scatter(Xe_test[:, 0][test0], Xe_test[:, 1][test0], color='cyan', alpha=alpha, label='test (non-manip)',
                   marker='2')
        ax.scatter(Xe_test[:, 0][test1], Xe_test[:, 1][test1], color='orange', alpha=alpha, label='test (manip)',
                   marker='x')

        ax.set_xlabel('tsne 0')
        ax.set_ylabel('tsne 1')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medifor1', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--plot_loss', action='store_true', default=False, help='plots loss of test instances.')
    parser.add_argument('--plot_all_sv', action='store_true', default=False, help='plots all support vectors.')
    parser.add_argument('--test_subset', default=100, type=int, help='num test instances to analyze.')
    parser.add_argument('--train_subset', default=10, type=int, help='train subset to use.')
    parser.add_argument('--test_type', default=None, type=str, help='type of test instances to inspect.')
    args = parser.parse_args()
    print(args)
    misclassification(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.plot_loss,
                      args.plot_all_sv, args.test_subset, args.train_subset, args.test_type)
