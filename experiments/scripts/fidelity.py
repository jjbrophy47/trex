"""
Experiment: How well does the linear model approximate the tree ensemble?
"""
import os
import sys
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner

import numpy as np

import trex
from utility import print_util
from utility import data_util
from utility import model_util
from utility import exp_util


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _get_knn_predictions(tree, knn_clf, X_test, X_test_alt, y_train):
    """
    Generate predictions for TEKNN.
    """

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

    res = {}
    res['tree'] = yhat_tree_test
    res['teknn'] = yhat_knn_test
    return res


def _get_trex_predictions(tree, explainer, data):
    """
    Generate predictions for TREX.
    """

    X_train, y_train, X_test, y_test = data
    multiclass = True if len(np.unique(y_train)) > 2 else False

    # tree ensemble predictions
    yhat_tree_test = tree.predict_proba(X_test)

    if not multiclass:
        yhat_tree_test = yhat_tree_test[:, 1].flatten()
    else:
        yhat_tree_test = yhat_tree_test.flatten()

    # linear model predictions
    if explainer.kernel_model == 'svm':
        yhat_trex_test = explainer.decision_function(X_test).flatten()
    else:
        yhat_trex_test = explainer.predict_proba(X_test)

        if not multiclass:
            yhat_trex_test = yhat_trex_test[:, 1].flatten()
        else:
            yhat_trex_test = yhat_trex_test.flatten()

    res = {}
    res['tree'] = yhat_tree_test
    res['trex'] = yhat_trex_test

    return res


def experiment(args, logger, out_dir, seed):

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=args.rs)

    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset,
                                                                 random_state=args.rs,
                                                                 data_dir=args.data_dir)

    # reduce train size
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train = int(X_train.shape[0] * args.train_frac)
        X_train, y_train = X_train[:n_train], y_train[:n_train]
    data = X_train, y_train, X_test, y_test

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('test instances: {}'.format(len(X_test)))
    logger.info('no. features: {}'.format(X_train.shape[1]))

    logger.info('no. trees: {:,}'.format(args.n_estimators))
    logger.info('max depth: {}'.format(args.max_depth))

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)

    if args.teknn:

        # transform data
        extractor = trex.TreeExtractor(tree, tree_kernel=args.tree_kernel)
        X_train_alt = extractor.fit_transform(X_train)
        X_test_alt = extractor.transform(X_test)
        train_label = y_train if args.true_label else tree.predict(X_train)

        # tune and train teknn
        start = time.time()
        logger.info('TE-KNN...')
        knn_clf = exp_util.tune_knn(tree, X_train, X_train_alt, train_label, args.val_frac,
                                    seed=seed, logger=logger)

        start = time.time()
        logger.info('generating predictions...')
        results = _get_knn_predictions(tree, knn_clf, X_test, X_test_alt, y_train)
        logger.info('time: {:.3f}s'.format(time.time() - start))

        results['n_neighbors'] = knn_clf.get_params()['n_neighbors']

        # save results
        np.save(os.path.join(out_dir, 'tree.npy'), results['tree'])
        np.save(os.path.join(out_dir, 'surrogate.npy'), results['teknn'])

    if args.trex:

        start = time.time()
        explainer = trex.TreeExplainer(tree, X_train, y_train,
                                       tree_kernel=args.tree_kernel,
                                       kernel_model=args.kernel_model,
                                       kernel_model_kernel=args.kernel_model_kernel,
                                       random_state=args.rs,
                                       logger=logger,
                                       true_label=not args.true_label,
                                       val_frac=args.val_frac)

        start = time.time()
        logger.info('generating predictions...')
        results = _get_trex_predictions(tree, explainer, data)
        logger.info('time: {:.3f}s'.format(time.time() - start))

        results['C'] = explainer.C

        # save data
        np.save(os.path.join(out_dir, 'tree.npy'), results['tree'])
        np.save(os.path.join(out_dir, 'surrogate.npy'.format(args.kernel_model)), results['trex'])


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, args.tree_type, args.tree_kernel)

    if args.trex:
        out_dir = os.path.join(out_dir, args.kernel_model)
    elif args.teknn:
        out_dir = os.path.join(out_dir, 'teknn')

    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    seed = args.rs
    logger.info('\nSeed: {}'.format(seed))
    experiment(args, logger, out_dir, seed=seed)
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/fidelity/', help='output directory.')

    parser.add_argument('--train_frac', type=float, default=1.0, help='amount of training data to use.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')

    parser.add_argument('--tree_type', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')

    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='type of encoding.')
    parser.add_argument('--true_label', action='store_true', default=False, help='Use true labels for explainer.')
    parser.add_argument('--kernel_model', type=str, default='klr', help='linear model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='Similarity kernel.')

    parser.add_argument('--trex', action='store_true', default=False, help='use TREX as surrogate model.')
    parser.add_argument('--teknn', action='store_true', default=False, help='Use KNN on top of TREX features.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()
    main(args)


# External API
class Args:
    dataset = 'churn'
    data_dir = 'data'
    out_dir = 'output/fidelity/'

    train_frac = 1.0
    val_frac = 0.1

    tree_type = 'cb'
    n_estimators = 100
    max_depth = None

    tree_kernel = 'tree_output'
    true_label = False
    kernel_model = 'klr'
    kernel_model_kernel = 'linear'

    trex = True
    teknn = False

    rs = 1
    verbose = 0
