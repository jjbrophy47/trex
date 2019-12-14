"""
Experiment: How well does the linear model approximate the tree ensemble?
"""
import os
import sys
import time
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner; TODO: remove this dependency

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier

import trex
from utility import model_util, data_util, print_util, exp_util


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _plot_knn_predictions(tree, knn_clf, data, knn_data, ax=None):

    X_train, y_train, X_test, y_test = data
    X_train_alt, y_train, X_test_alt, y_test = knn_data
    multiclass = True if len(np.unique(y_train)) > 2 else False

    # tree ensemble predictions
    yhat_tree_test = tree.predict_proba(X_test)
    yhat_knn_test = knn_clf.predict_proba(X_test_alt)

    if not multiclass:
        yhat_tree_test = yhat_tree_test[:, 1].flatten()
        yhat_knn_test = yhat_knn_test[:, 1].flatten()
    else:
        yhat_tree_test = yhat_tree_test.flatten()
        yhat_knn_test = yhat_knn_test.flatten()

    # compute correlation between tree probabilities and linear probabilities/decision values
    test_pear = np.corrcoef(yhat_tree_test, yhat_knn_test)[0][1]
    test_spear = spearmanr(yhat_tree_test, yhat_knn_test)[0]

    # plot results
    test_label = 'test={:.3f} (p), {:.3f} (s)'.format(test_pear, test_spear)

    ax.scatter(yhat_knn_test, yhat_tree_test, color='blue', label=test_label)

    res = {}
    res['tree'] = {'test': yhat_tree_test}
    res['ours'] = {'test': yhat_knn_test}
    return res


def _plot_predictions(tree, explainer, data, ax=None, use_sigmoid=False):

    X_train, y_train, X_test, y_test = data
    multiclass = True if len(np.unique(y_train)) > 2 else False

    # tree ensemble predictions
    yhat_tree_train = tree.predict_proba(X_train)
    yhat_tree_test = tree.predict_proba(X_test)

    if not multiclass:
        yhat_tree_train = yhat_tree_train[:, 1].flatten()
        yhat_tree_test = yhat_tree_test[:, 1].flatten()
    else:
        yhat_tree_train = yhat_tree_train.flatten()
        yhat_tree_test = yhat_tree_test.flatten()

    # linear model predictions
    if explainer.linear_model == 'svm':
        yhat_linear_train = explainer.decision_function(X_train).flatten()
        yhat_linear_test = explainer.decision_function(X_test).flatten()
    else:
        yhat_linear_train = explainer.predict_proba(X_train)
        yhat_linear_test = explainer.predict_proba(X_test)

        if not multiclass:
            yhat_linear_train = yhat_linear_train[:, 1].flatten()
            yhat_linear_test = yhat_linear_test[:, 1].flatten()
        else:
            yhat_linear_train = yhat_linear_train.flatten()
            yhat_linear_test = yhat_linear_test.flatten()

    if use_sigmoid and explainer.linear_model == 'svm':
        yhat_linear_train = _sigmoid(yhat_linear_train)
        yhat_linear_test = _sigmoid(yhat_linear_test)

    # compute correlation between tree probabilities and linear probabilities/decision values
    train_pear = np.corrcoef(yhat_tree_train, yhat_linear_train)[0][1]
    test_pear = np.corrcoef(yhat_tree_test, yhat_linear_test)[0][1]

    train_spear = spearmanr(yhat_tree_train, yhat_linear_train)[0]
    test_spear = spearmanr(yhat_tree_test, yhat_linear_test)[0]

    # plot results
    train_label = 'train={:.3f} (p), {:.3f} (s)'.format(train_pear, train_spear)
    test_label = 'test={:.3f} (p), {:.3f} (s)'.format(test_pear, test_spear)

    ax.scatter(yhat_linear_train, yhat_tree_train, color='blue', label=train_label)
    ax.scatter(yhat_linear_test, yhat_tree_test, color='cyan', label=test_label)

    res = {}
    res['tree'] = {'train': yhat_tree_train, 'test': yhat_tree_test}
    res['ours'] = {'train': yhat_linear_train, 'test': yhat_linear_test}

    return res


def fidelity(args, model='lgb', encoding='leaf_output', dataset='iris', n_estimators=100, C=0.1, random_state=69,
             true_label=False, data_dir='data', linear_model='lr', kernel='linear', use_sigmoid=False,
             out_dir='output/fidelity/', flip_frac=None, max_depth=None, knn=False, verbose=0,
             tune_knn=False, knn_neighbors=5, knn_weights='uniform'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)
    data = X_train, y_train, X_test, y_test

    # corrupt the data if specified
    if flip_frac is not None:
        y_train, noisy_ndx = data_util.flip_labels(y_train, k=flip_frac, random_state=random_state)
        noisy_ndx = np.array(sorted(noisy_ndx))
        print('num noisy labels: {}'.format(len(noisy_ndx)))

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)

    if knn:

        # write output to logs
        if tune_knn:
            setting = '{}_{}_knn_gs'.format(model, encoding)
        else:
            setting = '{}_{}_{}_{}'.format(model, encoding, knn_neighbors, knn_weights)

        out_dir = os.path.join(out_dir, dataset, setting)
        os.makedirs(out_dir, exist_ok=True)
        logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(dataset)))
        logger.info(args)

        extractor = trex.TreeExtractor(tree, encoding=encoding)
        X_train_alt = extractor.fit_transform(X_train)
        X_test_alt = extractor.transform(X_test)
        knn_data = X_train_alt, y_train, X_test_alt, y_test

        if tune_knn:
            knn_clf, params = exp_util.tune_knn(X_train_alt, y_train, tree, X_val_tree=X_test, X_val_knn=X_test_alt)
            logger.info('n_neighbors: {}, weights: {}'.format(params['n_neighbors'], params['weights']))
        else:
            logger.info('fitting knn...')
            knn_clf = KNeighborsClassifier(n_neighbors=knn_neighbors, weights=knn_weights).fit(X_train_alt, y_train)

        fig, ax = plt.subplots()

        start = time.time()
        logger.info('generating predictions...')
        results = _plot_knn_predictions(tree, knn_clf, data, knn_data, ax=ax)
        logger.info('time: {:.3f}s'.format(time.time() - start))

        ax.set_xlabel('knn')
        ax.set_ylabel('{}'.format(model))
        ax.set_title('{}, {}\n{}, {}'.format(dataset, model, knn_clf.n_neighbors, knn_clf.weights))
        ax.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(out_dir, 'fidelity.pdf'), format='pdf', bbox_inches='tight')
        np.save(os.path.join(out_dir, 'ours_test.npy'), results['ours']['test'])

    else:

        # make logger
        true_label_str = 'true_label' if true_label else ''
        sigmoid_str = 'sigmoid' if use_sigmoid else ''
        setting = '{}_{}_{}_{}_{}_{}_t{}_md{}'.format(model, linear_model, kernel, encoding, true_label_str,
                                                      sigmoid_str, n_estimators, max_depth)
        out_dir = os.path.join(out_dir, dataset, setting)
        os.makedirs(out_dir, exist_ok=True)
        logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(dataset)))
        logger.info(args)

        # plot fidelity
        fig, ax = plt.subplots()
        start = time.time()
        explainer = trex.TreeExplainer(tree, X_train, y_train, encoding=encoding, linear_model=linear_model, C=C,
                                       kernel=kernel, random_state=random_state, use_predicted_labels=not true_label)
        results = _plot_predictions(tree, explainer, data, ax=ax, use_sigmoid=use_sigmoid)
        logger.info('time: {:.3f}s'.format(time.time() - start))
        ax.set_xlabel('{}'.format(linear_model))
        ax.set_ylabel('{}'.format(model))
        ax.set_title('{}, {}\n{}, {}, {}, {}'.format(dataset, model, linear_model, kernel, encoding, true_label_str))
        ax.legend()
        plt.tight_layout()

        # save plot
        plt.savefig(os.path.join(out_dir, 'fidelity.pdf'), format='pdf', bbox_inches='tight')

        # save data
        np.save(os.path.join(out_dir, 'tree_train.npy'), results['tree']['train'])
        np.save(os.path.join(out_dir, 'tree_test.npy'), results['tree']['test'])
        np.save(os.path.join(out_dir, 'ours_train.npy'), results['ours']['train'])
        np.save(os.path.join(out_dir, 'ours_test.npy'), results['ours']['test'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--kernel', type=str, default='linear', help='Similarity kernel.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    parser.add_argument('--true_label', action='store_true', default=False, help='Use true labels for explainer.')
    parser.add_argument('--use_sigmoid', action='store_true', default=False, help='Run svm results through sigmoid.')
    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--gridsearch', action='store_true', default=False, help='Use gridsearch to tune KNN.')
    parser.add_argument('--knn_neighbors', type=int, default=5, help='Use KNN on top of TREX features.')
    parser.add_argument('--knn_weights', type=str, default='uniform', help='Use KNN on top of TREX features.')
    args = parser.parse_args()
    fidelity(args, model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
             random_state=args.rs, true_label=args.true_label, linear_model=args.linear_model, knn=args.knn,
             kernel=args.kernel, use_sigmoid=args.use_sigmoid, C=args.C, max_depth=args.max_depth,
             verbose=args.verbose, gridsearch=args.gridsearch, knn_neighbors=args.knn_neighbors,
             knn_weights=args.knn_weights)
