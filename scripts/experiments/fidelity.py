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

import matplotlib
matplotlib.use('Agg')  # allows plots when running with no display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for TREX
import trex
import util


def experiment(args, out_dir, logger):

    # get data
    data = util.get_data(args.dataset,
                         data_dir=args.data_dir,
                         preprocessing=args.preprocessing)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # randomly select a subset of test instances to evaluate with
    rng = np.random.default_rng(args.rs)
    indices = rng.choice(X_test.shape[0], size=args.n_test, replace=False)
    X_test, y_test = X_test[indices], y_test[indices]

    logger.info('\ntrain instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # get tree-ensemble
    model = util.get_model(args.model,
                           n_estimators=args.n_estimators,
                           max_depth=args.max_depth,
                           random_state=args.rs,
                           cat_indices=cat_indices)

    logger.info('\nno. trees: {:,}'.format(args.n_estimators))
    logger.info('max depth: {}'.format(args.max_depth))

    # train a tree ensemble
    start = time.time()
    model = model.fit(X_train, y_train)
    logger.info('\nfitting tree ensemble...{:.3f}s'.format(time.time() - start))

    # make predictions
    model_proba = model.predict_proba(X_test)[:, 1]

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
                                       weighted=args.weighted,
                                       random_state=args.rs,
                                       logger=logger)
        train_time = time.time() - start
        logger.info('fitting surrogate model...{:.3f}s'.format(train_time))

        # make predictions
        start = time.time()
        surrogate_proba = surrogate.predict_proba(X_test)[:, 1]
        logger.info('prediction time...{:.3f}s'.format(time.time() - start))

        # get no. features in the tree kernel space
        n_features_alt = surrogate.n_features_alt_

    # KNN operating on a transformed feature space
    elif args.surrogate == 'knn':

        # initialie feature extractor
        feature_extractor = trex.TreeExtractor(model, tree_kernel=args.tree_kernel)

        # transform the training data using the tree extractor
        start = time.time()
        X_train_alt = feature_extractor.transform(X_train)
        end = time.time() - start
        logger.info('transforming train data using {} kernel...{:.3f}s'.format(args.tree_kernel, end))

        # transform the test data using the tree extractor
        start = time.time()
        X_test_alt = feature_extractor.transform(X_test)
        end = time.time() - start
        logger.info('transforming test data using {} kernel...{:.3f}s'.format(args.tree_kernel, end))
        logger.info('no. features after transformation: {:,}'.format(X_train_alt.shape[1]))

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
                                              weighted=args.weighted,
                                              logger=logger)
        train_time = time.time() - start

        # make predictions
        start = time.time()
        surrogate_proba = surrogate.predict_proba(X_test_alt)[:, 1]
        logger.info('prediction time...{:.3f}s'.format(time.time() - start))

        # get no. features in the alternate tree kernel space
        n_features_alt = X_train_alt.shape[1]

    else:
        raise ValueError('surrogate {} unknown!'.format(args.surrogate))

    # save results
    result = {}
    result['model'] = args.model
    result['n_estimators'] = args.n_estimators
    result['max_depth'] = args.max_depth
    result['surrogate'] = args.surrogate
    result['tree_kernel'] = args.tree_kernel
    result['train_time'] = train_time
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['tune_frac'] = args.tune_frac
    result['model_proba'] = model_proba
    result['surrogate_proba'] = surrogate_proba
    result['pearson'] = pearsonr(model_proba, surrogate_proba)[0]
    result['spearman'] = spearmanr(model_proba, surrogate_proba)[0]
    result['mse'] = mean_squared_error(model_proba, surrogate_proba)
    result['n_features_alt'] = n_features_alt
    np.save(os.path.join(out_dir, 'results.npy'), result)
    logger.info('\nresults:\n{}'.format(result))

    # save scatter
    fig, ax = plt.subplots()
    ax.scatter(result['surrogate_proba'], result['model_proba'])
    ax.set_xlabel('Surrogate prob.')
    ax.set_ylabel('Tree-ensemble prob.')
    plt.savefig(os.path.join(out_dir, 'scatter.pdf'))

    logger.info('\nsaving results to {}/...'.format(os.path.join(out_dir)))


def main(args):

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.preprocessing,
                           args.surrogate,
                           args.tree_kernel,
                           args.metric,
                           'rs_{}'.format(args.rs))

    # create output directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--preprocessing', type=str, default='categorical', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/fidelity/', help='output directory.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth in tree ensemble.')

    # Surrogate settings
    parser.add_argument('--surrogate', type=str, default='klr', help='klr, svm, or knn.')
    parser.add_argument('--kernel_model', type=str, default='klr', help='klr or svm.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of tree feature extraction.')
    parser.add_argument('--tune_frac', type=float, default=0.1, help='fraction of training data to use for tuning.')
    parser.add_argument('--metric', type=str, default='mse', help='pearson, spearman, or mse.')
    parser.add_argument('--weighted', type=int, default=0, help='If 1, train surrogate on weighted train instances.')

    # Experiment settings
    parser.add_argument('--n_test', type=int, default=1000, help='no. test samples to test fidelity.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')

    args = parser.parse_args()
    main(args)
