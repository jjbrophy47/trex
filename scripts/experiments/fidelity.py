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

    # start timer
    begin = time.time()

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

    # model predictions
    model_proba = model.predict_proba(X_test)[:, 1]

    # used if no tuning is done on the surrogate model
    params = {'C': args.C,
              'n_neighbors': args.n_neighbors,
              'tree_kernel': args.tree_kernel}

    # train and predict with using a surrogate model
    start = time.time()
    surrogate = trex.train_surrogate(model=model,
                                     surrogate=args.surrogate,
                                     X_train=X_train,
                                     y_train=y_train,
                                     val_frac=args.tune_frac,
                                     metric=args.metric,
                                     seed=args.rs,
                                     params=params,
                                     logger=logger)
    train_time = time.time() - start

    # make predictions
    start = time.time()
    surrogate_proba = surrogate.predict_proba(X_test)[:, 1]
    logger.info('prediction time...{:.3f}s'.format(time.time() - start))

    # save results
    result = {}
    result['model'] = args.model
    result['model_n_estimators'] = args.n_estimators
    result['model_max_depth'] = args.max_depth
    result['model_proba'] = model_proba
    result['surrogate'] = args.surrogate
    result['surrogate_n_features'] = surrogate.n_features_alt_
    result['surrogate_tree_kernel'] = surrogate.tree_kernel_
    result['surrogate_C'] = surrogate.C if hasattr(surrogate, 'C') else None
    result['surrogate_n_neighbors'] = surrogate.n_neighbors if hasattr(surrogate, 'n_neighbors') else None
    result['surrogate_train_time'] = train_time
    result['surrogate_proba'] = surrogate_proba
    result['total_time'] = time.time() - begin
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['tune_frac'] = args.tune_frac
    result['pearson'] = pearsonr(model_proba, surrogate_proba)[0]
    result['spearman'] = spearmanr(model_proba, surrogate_proba)[0]
    result['mse'] = mean_squared_error(model_proba, surrogate_proba)
    np.save(os.path.join(out_dir, 'results.npy'), result)
    logger.info('\nresults:\n{}'.format(result))

    # save scatter
    fig, ax = plt.subplots()
    ax.scatter(result['surrogate_proba'], result['model_proba'])
    ax.set_xlabel('Surrogate prob.')
    ax.set_ylabel('Tree-ensemble prob.')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig(os.path.join(out_dir, 'scatter.pdf'))

    logger.info('\nsaving results to {}/...'.format(os.path.join(out_dir)))


def main(args):

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.preprocessing,
                           args.surrogate,
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
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/fidelity/', help='output directory.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth in tree ensemble.')

    # Surrogate settings
    parser.add_argument('--surrogate', type=str, default='klr', help='klr, svm, or knn.')
    parser.add_argument('--tune_frac', type=float, default=0.1, help='fraction of training data to use for tuning.')
    parser.add_argument('--metric', type=str, default='mse', help='pearson, spearman, or mse.')

    # No tuning settings
    parser.add_argument('--C', type=float, default=0.1, help='penalty parameters for KLR or SVM.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='no. neighbors to use for KNN.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    # Experiment settings
    parser.add_argument('--n_test', type=int, default=1000, help='no. test samples to test fidelity.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')

    args = parser.parse_args()
    main(args)
