"""
Experiment: How well does the linear model approximate the tree ensemble?
"""
import os
import sys
import time
import argparse
import resource
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import trex
from utility import print_util
from utility import data_util
from utility import model_util
from utility import exp_util


def _get_knn_predictions(tree, knn_clf, X_test, X_test_alt,
                         y_train, pred_size=None, out_dir=None,
                         logger=None):
    """
    Generate predictions for TEKNN.
    """

    multiclass = True if len(np.unique(y_train)) > 2 else False

    if pred_size is not None:
        pred_indices = np.random.choice(X_test.shape[0],
                                        size=X_test.shape[0],
                                        replace=False)

        tree_preds = []
        knn_preds = []

        start = time.time()
        for i in range(0, X_test.shape[0], pred_size):

            sub_indices = pred_indices[i: i + pred_size]
            X_test_sub = X_test[sub_indices]
            X_test_sub_alt = X_test_alt[sub_indices]

            tree_pred = tree.predict_proba(X_test_sub)
            knn_pred = knn_clf.predict_proba(X_test_sub_alt)

            if not multiclass:
                tree_preds.append(tree_pred[:, 1])
                knn_preds.append(knn_pred[:, 1])

            else:
                tree_preds.append(tree_pred.flatten())
                knn_preds.append(knn_pred.flatten())

            tree_result = np.concatenate(tree_preds)
            knn_result = np.concatenate(knn_preds)

            # save results
            log_time = time.time() - start
            s = 'saving {} predictions, cumulative time {:.3f}s'
            logger.info(s.format(len(tree_result), log_time))

            # save results
            np.save(os.path.join(out_dir, 'tree.npy'), tree_result)
            np.save(os.path.join(out_dir, 'surrogate.npy'), knn_result)

        return None

    else:

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

    # kernel model predictions
    yhat_trex_test = explainer.predict_proba(X_test)

    if not multiclass:
        yhat_trex_test = yhat_trex_test[:, 1]
    yhat_trex_test = yhat_trex_test.flatten()

    res = {}
    res['tree'] = yhat_tree_test
    res['trex'] = yhat_trex_test

    return res


def experiment(args, out_dir, logger):

    # get data
    data = data_util.get_data(args.dataset,
                              data_dir=args.data_dir,
                              processing_dir=args.processing_dir)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # randomly select a fraction of the train data to use for tuning
    if args.tune_frac < 1.0 and args.tune_frac > 0.0:
        n_tune = int(X_train.shape[0] * args.tune_frac)
        X_val, y_val = X_train[:n_tune], y_train[:n_tune]

    # randomly select a subset of test instances to evaluate with
    rng = np.random.default_rng(args.rs)
    indices = rng.choice(X_test.shape[0], size=args.n_test, replace=False)
    X_test, y_test = X_test[indices], y_test[indices]

    logger.info('train instances: {}'.format(X_train.shape[0]))
    logger.info('test instances: {}'.format(X_val.shape[0]))
    logger.info('no. features: {}'.format(X_train.shape[1]))
    logger.info('no. tune instances: {}'.format(X_val.shape[0]))

    # get tree-ensemble
    clf = model_util.get_classifier(args.model,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=args.rs,
                                    cat_indices=cat_indices)

    logger.info('no. trees: {:,}'.format(args.n_estimators))
    logger.info('max depth: {}'.format(args.max_depth))

    # train a tree ensemble
    logger.info('fitting tree ensemble...')
    model = clf.fit(X_train, y_train)

    # train surrogate model
    if 'trex' in args.surrogate:

        start = time.time()
        surrogate = trex.TreeExplainer(model,
                                       X_train,
                                       y_train,
                                       kernel_model=args.kernel_model,
                                       tree_kernel=args.tree_kernel,
                                       random_state=args.rs,
                                       logger=logger)

        start = time.time()
        logger.info('generating predictions...')
        results = _get_trex_predictions(model, explainer, data)
        logger.info('time: {:.3f}s'.format(time.time() - start))

        results['C'] = explainer.C

    # KNN operating on a transformed feature space
    elif 'teknn' == args.surrogate:

        # initialie feature extractor
        feature_extractor = trex.TreeExtractor(model, tree_kernel=args.tree_kernel)

        # transform the training data using the tree extractor
        start = time.time()
        X_train_alt = feature_extractor.fit_transform(X_train)
        end = time.time() - start
        logger.info('transforming train data using {} kernel...'.format(args.tree_kernel, end))

        # transform the validation data using the tree extractor
        start = time.time()
        X_train_alt = feature_extractor.fit_transform(X_val)
        end = time.time() - start
        logger.info('transforming train data using {} kernel...'.format(args.tree_kernel, end))

        # transform the test data using the tree extractor
        start = time.time()
        X_test_alt = feature_extractor.transform(X_test)
        logger.info('transforming test data using {} kernel...'.format(args.tree_kernel, end))
        end = time.time() - start

        surrogate = train_surrogate(args.surrogate, X_train, y_train, X_val, y_val, rng, logger)

    # make predictions using the tree-ensemble and the surrogate
    model_proba = model.predict_proba(X_test)[:, 1]
    surrogate_proba = surrogate.predict_proba(X_test)[:, 1]

    # save results
    result = {}
    result['model'] = args.model
    result['surrogate'] = args.surrogate
    # result['tune_time'] = train_time
    # result['train_time'] = train_time
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['tune_frac'] = args.tune_frac
    result['model_proba'] = model_proba
    result['surrogate_proba'] = surrogate_proba
    np.save(os.path.join(out_dir, 'results.npy'), result)

    if args.teknn:

        # transform data
        feature_extractor = trex.TreeExtractor(tree, tree_kernel=args.tree_kernel)

        logger.info('transforming training data...')
        X_train_alt = feature_extractor.fit_transform(X_train)

        logger.info('transforming test data...')
        X_test_alt = feature_extractor.transform(X_test)

        train_label = y_train if args.true_label else tree.predict(X_train)

        # tune and train teknn
        start = time.time()
        logger.info('TE-KNN...')
        if args.k:
            knn_clf = KNeighborsClassifier(n_neighbors=args.k, weights='uniform')
            knn_clf = knn_clf.fit(X_train_alt, y_train)
        else:
            knn_clf = exp_util.tune_knn(tree, X_train, X_train_alt, train_label, args.val_frac,
                                        seed=seed, logger=logger)

        start = time.time()
        logger.info('generating predictions...')
        results = _get_knn_predictions(tree, knn_clf, X_test, X_test_alt, y_train,
                                       pred_size=args.pred_size, out_dir=out_dir,
                                       logger=logger)
        logger.info('time: {:.3f}s'.format(time.time() - start))

        # save results
        if results:
            results['n_neighbors'] = knn_clf.get_params()['n_neighbors']
            np.save(os.path.join(out_dir, 'tree.npy'), results['tree'])
            np.save(os.path.join(out_dir, 'surrogate.npy'), results['teknn'])

    if args.trex:

        start = time.time()
        explainer = trex.TreeExplainer(tree, X_train, y_train,
                                       tree_kernel=args.tree_kernel,
                                       kernel_model=args.kernel_model,
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

    out_dir = os.path.join(args.out_dir,
                           dataset,
                           args.model,
                           args.surrogate,
                           args.tree_kernel)

    # create output directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    print_util.clear_dir(out_dir)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # write everything printed to stdout to this log file
    logfile, stdout, stderr = print_util.stdout_stderr_to_log(os.path.join(out_dir, 'log+.txt'))

    # run experiment
    experiment(args, out_dir, logger)

    # restore original stdout and stderr settings
    print_util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/fidelity/', help='output directory.')

    # Data settings
    # parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training data to use for tuning.')
    # parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')

    # Surrogate settings
    parser.add_argument('--surrogate', type=str, default='trex', help='trex_lr, trex_svm, or teknn.')
    parser.add_argument('--kernel_model', type=str, default='klr', help='linear model to use.')
    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='type of encoding.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training data to use for tuning.')
    # parser.add_argument('--true_label', action='store_true', default=False, help='Use true labels for explainer.')

    # method settings
    # parser.add_argument('--surrogate', type=str, default='trex', help='trex or teknn.')
    # parser.add_argument('--k', type=int, default=None, help='no. neighbors.')

    # Experiment settings
    parser.add_argument('--n_test', type=int, default=1000, help='chunk size.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)
