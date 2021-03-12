"""
Experiment:
    1) Record time to train / initialize explainer.
    2) Record time to compute influence of ALL training instances on a SINGLE test instance.
"""
import os
import sys
import time
import uuid
import shutil
import signal
import argparse
import resource
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
from copy import deepcopy
from datetime import datetime

import numpy as np
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility
import trex
import util
from baselines.influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from baselines.maple.MAPLE import MAPLE


class timeout:
    """
    Timeout class to throw a TimeoutError if a piece of code runs for too long.
    """
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def trex_method(args, model, test_ndx, X_train, y_train, X_test,
                logger=None):
    """
    Generate an training instance attributions for a test instance.
    """

    # train surrogate model
    start = time.time()
    params = {'C': args.C, 'n_neighbors': args.n_neighbors, 'tree_kernel': args.tree_kernel}
    surrogate = trex.train_surrogate(model=model,
                                     surrogate=args.method,
                                     X_train=X_train,
                                     y_train=y_train,
                                     val_frac=args.tune_frac,
                                     metric=args.metric,
                                     seed=args.rs,
                                     params=params,
                                     logger=logger)
    train_time = time.time() - start

    # compute influential training instances on the test instance
    start = time.time()
    surrogate.compute_attributions(X_test[test_ndx])
    test_time = time.time() - start

    # result object
    result = {'train_time': train_time, 'test_time': test_time}

    return result


def leaf_influence_method(args, model, test_ndx, X_train, y_train,
                          X_test, y_test, k=-1, update_set='AllPoints',
                          frac_progress_update=0.1, logger=None):
    """
    Computes the influence on each test instance if train
    instance i were upweighted/removed.

    NOTE: This uses the FastLeafInfluence (k=0) method by Sharchilev et al.
    NOTE: requires the label for the test instance.
    """

    # LeafInfluence settings
    if 'fast' in args.method:
        k = 0
        update_set = 'SinglePoint'

    # initialize Leaf Influence
    start = time.time()

    # save CatBoost model
    temp_dir = os.path.join('.catboost_info', 'leaf_influence_{}'.format(str(uuid.uuid4())))
    temp_fp = os.path.join(temp_dir, 'cb.json')
    os.makedirs(temp_dir, exist_ok=True)
    model.save_model(temp_fp, format='json')

    explainer = CBLeafInfluenceEnsemble(temp_fp,
                                        X_train,
                                        y_train,
                                        k=k,
                                        learning_rate=model.learning_rate_,
                                        update_set=update_set)
    train_time = time.time() - start

    if logger:
        logger.info('\ncomputing influence of each training instance on the test loss...')

    # compute influence of each training instance on the test instance
    with timeout(seconds=args.max_time):
        try:
            start = time.time()

            contributions = []
            buf = deepcopy(explainer)

            for i in range(X_train.shape[0]):
                explainer.fit(removed_point_idx=i, destination_model=buf)
                contributions.append(buf.loss_derivative(X_test[test_ndx], y_test[test_ndx])[0])

                # display progress
                if logger and i % int(X_train.shape[0] * frac_progress_update) == 0:
                    elapsed = time.time() - start
                    train_frac_complete = i / X_train.shape[0] * 100
                    logger.info('Train {:.1f}%...{:.3f}s'.format(train_frac_complete, elapsed))

            contributions = np.array(contributions)
            test_time = time.time() - start

        except TimeoutError:
            if logger:
                logger.info('Leaf Influence test time exceeded!')
                exit(0)

    # clean up
    shutil.rmtree(temp_dir)

    # result object
    result = {'train_time': train_time, 'test_time': test_time}

    return result


def maple_method(args, model, test_ndx, X_train, y_train, X_test,
                 dstump=False, logger=None):
    """
    Produces a train weight distribution for a single test instance.
    """
    train_label = model.predict(X_train)

    with timeout(seconds=args.max_time):
        try:
            start = time.time()
            maple = MAPLE(X_train, train_label, X_train, train_label, dstump=dstump)
            train_time = time.time() - start

        except TimeoutError:
            if logger:
                logger.info('MAPLE fine-tuning exceeded!')
            exit(0)

    # compute "local training distribution" for the test instance
    start = time.time()
    maple.explain(X_test[test_ndx]) if dstump else maple.get_weights(X_test[test_ndx])
    test_time = time.time() - start

    # result object
    result = {'train_time': train_time, 'test_time': test_time}

    return result


def teknn_method(args, model, test_ndx, X_train, y_train, X_test,
                 logger=None):
    """
    KNN surrogate method that retrieves its k nearest neighbors as most influential.
    """
    with timeout(seconds=args.max_time):
        try:

            # train surrogate model
            start = time.time()
            params = {'C': args.C, 'n_neighbors': args.n_neighbors, 'tree_kernel': args.tree_kernel}
            surrogate = trex.train_surrogate(model=model,
                                             surrogate=args.method,
                                             X_train=X_train,
                                             y_train=y_train,
                                             val_frac=args.tune_frac,
                                             metric=args.metric,
                                             seed=args.rs,
                                             params=params,
                                             logger=logger)

            train_time = time.time() - start

        except TimeoutError:
            if logger:
                logger.info('TEKNN fine-tuning exceeded!')
            exit(0)

    # retrieve k nearest neighbors as most influential to the test instance
    start = time.time()
    surrogate.compute_attributions(X_test[test_ndx])
    test_time = time.time() - start

    # result object
    result = {'train_time': train_time, 'test_time': test_time}

    return result


def experiment(args, logger, out_dir):
    """
    Main method that trains a tree ensemble, then compares the
    runtime of different methods to explain a single test instance.
    """

    # start timer
    begin = time.time()

    # create random number generator
    rng = np.random.default_rng(args.rs)

    # get data
    data = util.get_data(args.dataset,
                         data_dir=args.data_dir,
                         preprocessing=args.preprocessing)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # get tree-ensemble
    clf = util.get_model(args.model,
                         n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         random_state=args.rs,
                         cat_indices=cat_indices)

    logger.info('\nno. train instances: {:,}'.format(X_train.shape[0]))
    logger.info('no. test instances: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}\n'.format(X_train.shape[1]))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    util.performance(model, X_train, y_train, logger=logger, name='Train')
    util.performance(model, X_test, y_test, logger=logger, name='Test')

    # randomly pick a test instances to explain
    test_ndx = rng.choice(y_test.shape[0], size=1, replace=False)

    # TREX
    if 'klr' in args.method or 'svm' in args.method:
        result = trex_method(args, model, test_ndx, X_train, y_train, X_test, logger=logger)

    # Leaf Influence
    elif 'leaf_influence' in args.method and args.model == 'cb':
        result = leaf_influence_method(args, model, test_ndx, X_train, y_train, X_test, y_test, logger=logger)

    # MAPLE
    elif args.method == 'maple':
        result = maple_method(args, model, test_ndx, X_train, y_train, X_test, logger=logger)

    # TEKNN
    elif 'knn' in args.method:
        result = teknn_method(args, model, test_ndx, X_train, y_train, X_test, logger=logger)

    else:
        raise ValueError('method {} unknown!'.format(args.method))

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['total_time'] = time.time() - begin
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # display results
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))


def main(args):

    # make logger
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.preprocessing,
                           args.method,
                           'rs_{}'.format(args.rs))

    # create output directory
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to evaluate.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/runtime/', help='output directory.')

    # Data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='fraction of data for training.')
    parser.add_argument('--tune_frac', type=float, default=0.0, help='fraction of train data for validation.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble.')
    parser.add_argument('--n_estimators', type=int, default=10, help='no. trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='max. depth in tree ensemble.')

    # Method settings
    parser.add_argument('--method', type=str, default='klr-leaf_output', help='influence method.')
    parser.add_argument('--metric', type=str, default='mse', help='surrogate tuning metric.')

    # No tuning settings
    parser.add_argument('--C', type=float, default=0.1, help='penalty parameters for KLR or SVM.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='no. neighbors to use for KNN.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--max_time', type=int, default=259200, help='max. experiment time in seconds.')

    # Additional Settings
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)
