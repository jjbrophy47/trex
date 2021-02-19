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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import trex
from utility import print_util
from utility import data_util
from utility import model_util


def experiment(args, out_dir, logger):

    # get data
    data = data_util.get_data(args.dataset,
                              data_dir=args.data_dir,
                              processing_dir=args.processing_dir)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # randomly select a subset of test instances to evaluate with
    rng = np.random.default_rng(args.rs)
    indices = rng.choice(X_test.shape[0], size=args.n_test, replace=False)
    X_test, y_test = X_test[indices], y_test[indices]

    logger.info('\ntrain instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # get tree-ensemble
    model = model_util.get_model(args.model,
                                 n_estimators=args.n_estimators,
                                 max_depth=args.max_depth,
                                 random_state=args.rs,
                                 cat_indices=cat_indices)

    logger.info('\nno. trees: {:,}'.format(args.n_estimators))
    logger.info('max depth: {}'.format(args.max_depth))

    # train a tree ensemble
    logger.info('\nfitting tree ensemble...')
    model = model.fit(X_train, y_train)

    # make predictions
    model_proba = model.predict_proba(X_test)[:, 1]

    # record train time
    start = time.time()

    # train and predict with using a surrogate model
    if args.surrogate in ['klr', 'svm']:

        start = time.time()
        surrogate = trex.TreeExplainer(model,
                                       X_train,
                                       y_train,
                                       kernel_model=args.kernel_model,
                                       tree_kernel=args.tree_kernel,
                                       val_frac=args.tune_frac,
                                       metric=args.metric,
                                       random_state=args.rs,
                                       logger=logger)
        train_time = time.time() - start
        logger.info('train time...{:.3f}s'.format(train_time))

        # make predictions
        start = time.time()
        surrogate_proba = surrogate.predict_proba(X_test)[:, 1]
        logger.info('prediction time...{:.3f}s'.format(time.time() - start))

    # KNN operating on a transformed feature space
    elif args.surrogate == 'knn':

        # initialie feature extractor
        feature_extractor = trex.TreeExtractor(model, tree_kernel=args.tree_kernel)

        # transform the training data using the tree extractor
        start = time.time()
        X_train_alt = feature_extractor.fit_transform(X_train)
        end = time.time() - start
        logger.info('transforming train data using {} kernel...{:.3f}s'.format(args.tree_kernel, end))

        # transform the test data using the tree extractor
        start = time.time()
        X_test_alt = feature_extractor.transform(X_test)
        logger.info('transforming test data using {} kernel...{:.3f}s'.format(args.tree_kernel, end))
        end = time.time() - start

        # train surrogate model
        start = time.time()
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]}
        surrogate = trex.util.train_surrogate(model,
                                              args.surrogate,
                                              param_grid,
                                              X_train,
                                              X_train_alt,
                                              y_train,
                                              val_frac=args.tune_frac,
                                              metric=args.metric,
                                              seed=args.rs,
                                              logger=logger)
        train_time = time.time() - start

        # make predictions
        start = time.time()
        surrogate_proba = surrogate.predict_proba(X_test_alt)[:, 1]
        logger.info('prediction time...{:.3f}s'.format(time.time() - start))

    else:
        raise ValueError('surrogate {} unknown!'.format(args.surrogate))

    # save results
    result = {}
    result['model'] = args.model
    result['surrogate'] = args.surrogate
    result['train_time'] = train_time
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['tune_frac'] = args.tune_frac
    result['model_proba'] = model_proba
    result['surrogate_proba'] = surrogate_proba
    result['pearson'] = pearsonr(model_proba, surrogate_proba)[0]
    result['spearman'] = spearmanr(model_proba, surrogate_proba)[0]
    result['mse'] = mean_squared_error(model_proba, surrogate_proba)
    np.save(os.path.join(out_dir, 'results.npy'), result)

    logger.info('\nresults:\n{}'.format(result))
    logger.info('saving results to {}...'.format(os.path.join(out_dir, 'results.npy')))


def main(args):

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
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
    parser.add_argument('--processing_dir', type=str, default='standard', help='processing type.')
    parser.add_argument('--out_dir', type=str, default='output/fidelity/', help='output directory.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')

    # Surrogate settings
    parser.add_argument('--surrogate', type=str, default='klr', help='klr, svm, or knn.')
    parser.add_argument('--kernel_model', type=str, default='klr', help='klr or svm.')
    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='type of tree feature extraction.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of training data to use for tuning.')
    parser.add_argument('--metric', type=str, default='pearson', help='pearson, spearman, or mse.')

    # Experiment settings
    parser.add_argument('--n_test', type=int, default=1000, help='no. test samples to test fidelity.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)
