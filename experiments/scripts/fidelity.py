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
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import trex
from utility import model_util, data_util, print_util, exp_util


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _plot_knn_predictions(tree, knn_clf, X_test, X_test_alt, y_train, ax=None):

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


def main(args):

    # create directory
    if args.knn:
        setting = '{}_teknn_{}'.format(args.tree_type, args.tree_kernel)
    else:
        setting = '{}_{}_{}_{}'.format(args.tree_type, args.kernel_model,
                                       args.kernel_model_kernel,
                                       args.tree_kernel)

    # write output to logs
    out_dir = os.path.join(args.out_dir, args.dataset, setting)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)
    logger.info(time.ctime(time.time()))

    # get model and data
    clf = model_util.get_classifier(args.tree_type, n_estimators=args.n_estimators, max_depth=args.max_depth,
                                    random_state=args.rs)
    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=args.rs,
                                                                 data_dir=args.data_dir)
    data = X_train, y_train, X_test, y_test

    # corrupt the data if specified
    if args.flip_frac is not None:
        y_train, noisy_ndx = data_util.flip_labels(y_train, k=args.flip_frac, random_state=args.rs)
        noisy_ndx = np.array(sorted(noisy_ndx))
        print('num noisy labels: {}'.format(len(noisy_ndx)))

    # use part of the test data as validation data
    X_val = X_test.copy()
    if args.val_frac < 1.0 and args.val_frac > 0.0:
        X_val = X_val[int(X_val.shape[0] * args.val_frac):]

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('val instances: {}'.format(len(X_val)))
    logger.info('test instances: {}'.format(len(X_test)))
    logger.info('no. features: {}'.format(X_train.shape[1]))

    logger.info('no. trees: {:,}'.format(args.n_estimators))
    logger.info('max depth: {}'.format(args.max_depth))

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)

    if args.knn:

        # transform data
        extractor = trex.TreeExtractor(tree, tree_kernel=args.tree_kernel)
        X_train_alt = extractor.fit_transform(X_train)
        X_test_alt = extractor.transform(X_test)
        X_val_alt = extractor.transform(X_val)
        train_label = y_train if args.true_label else tree.predict(X_train)

        # tune and train teknn
        start = time.time()
        logger.info('tuning TE-KNN...')
        knn_clf, params = exp_util.tune_knn(X_train_alt, train_label, tree, X_val, X_val_alt, logger=logger)
        logger.info('n_neighbors: {}, weights: {}'.format(params['n_neighbors'], params['weights']))
        logger.info('time: {:.3f}s'.format(time.time() - start))

        fig, ax = plt.subplots()

        start = time.time()
        logger.info('generating predictions...')
        results = _plot_knn_predictions(tree, knn_clf, X_test, X_test_alt, y_train, ax=ax)
        logger.info('time: {:.3f}s'.format(time.time() - start))

        ax.set_xlabel('knn')
        ax.set_ylabel('{}'.format(args.tree_type))
        ax.set_title('{}, {}\n{}, {}'.format(args.dataset, args.tree_type, knn_clf.n_neighbors, knn_clf.weights))
        ax.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(out_dir, 'fidelity.pdf'), format='pdf', bbox_inches='tight')
        np.save(os.path.join(out_dir, 'ours_test.npy'), results['ours']['test'])
        np.save(os.path.join(out_dir, 'tree_test.npy'), results['tree']['test'])

    else:

        # plot fidelity
        fig, ax = plt.subplots()
        start = time.time()
        logger.info('tuning TREX-{}...'.format(args.kernel_model))
        explainer = trex.TreeExplainer(tree, X_train, y_train,
                                       tree_kernel=args.tree_kernel,
                                       kernel_model=args.kernel_model,
                                       C=args.C, kernel=args.kernel_model_kernel,
                                       random_state=args.rs,
                                       logger=logger,
                                       use_predicted_labels=not args.true_label,
                                       X_val=X_val)
        logger.info('C: {}'.format(explainer.C))
        logger.info('time: {:.3f}s'.format(time.time() - start))

        logger.info('generating predictions...')
        results = _plot_predictions(tree, explainer, data, ax=ax, use_sigmoid=args.use_sigmoid)
        logger.info('time: {:.3f}s'.format(time.time() - start))
        ax.set_xlabel('TREX-{}'.format(args.kernel_model.upper()))
        ax.set_ylabel('{}'.format(args.tree_type.upper()))
        ax.set_title('Dataset: {}, Tree kernel: {}'.format(args.dataset.capitalize(),
                                                           args.tree_kernel))
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
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/fidelity/', help='output directory.')

    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')
    parser.add_argument('--flip_frac', type=float, default=None, help='Fraction of train labels to flip.')

    parser.add_argument('--tree_type', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')

    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--true_label', action='store_true', default=False, help='Use true labels for explainer.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='Similarity kernel.')
    parser.add_argument('--use_sigmoid', action='store_true', default=False, help='Run svm results through sigmoid.')

    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')

    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=1, help='for reproducibility.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()
    main(args)


# External API
class Args:
    dataset = 'churn'
    data_dir = 'data'
    out_dir = 'output/fidelity/'

    val_frac = 0.1
    flip_frac = None

    tree_type = 'cb'
    n_estimators = 100
    max_depth = None
    C = 0.1

    tree_kernel = 'leaf_output'
    true_label = False
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'
    use_sigmoid = False

    knn = False

    rs = 1
    verbose = 0
