"""
Experiment: Compare runtimes for explaining a single test instance for different methods.
"""
import time
import argparse
import os
import sys
import signal
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import numpy as np
from sklearn.base import clone
from maple import MAPLE

import trex
from utility import model_util
from utility import data_util
from utility import print_util
from utility import exp_util

ONE_DAY = 43200  # number of seconds in 12 hours
maple_limit_reached = False
teknn_limit_reached = False


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


def _trex_method(test_ndx, X_test, model, X_train, y_train,
                 args, seed, logger=None):
    """
    Explains the predictions of each test instance.
    """

    # get validation
    X_val = exp_util.get_val_data(X_train, args.val_frac, seed)

    start = time.time()
    explainer = trex.TreeExplainer(model, X_train, y_train,
                                   tree_kernel=args.tree_kernel,
                                   random_state=seed,
                                   true_label=args.true_label,
                                   kernel_model=args.kernel_model,
                                   kernel_model_kernel=args.kernel_model_kernel,
                                   verbose=args.verbose,
                                   X_val=X_val,
                                   logger=logger)
    fine_tune = time.time() - start

    start = time.time()
    explainer.explain(X_test[test_ndx].reshape(1, -1))
    test_time = time.time() - start

    return fine_tune, test_time


def _influence_method(model, test_ndx, X_train, y_train, X_test, y_test, inf_k):
    """
    Computes the influence on each test instance if train instance i were upweighted/removed.
    This uses the fastleafinfluence method by Sharchilev et al.
    """

    start = time.time()
    leaf_influence = exp_util.get_influence_explainer(model, X_train, y_train, inf_k)
    fine_tune = time.time() - start

    start = time.time()
    exp_util.influence_explain_instance(leaf_influence, test_ndx, X_train, X_test, y_test)
    test_time = time.time() - start

    return fine_tune, test_time


def _maple_method(model, test_ndx, X_train, y_train, X_test, y_test, dstump=False, logger=None):
    """
    Produces a train weight distribution for a single test instance.
    """
    with timeout(seconds=ONE_DAY):
        try:
            start = time.time()
            maple = MAPLE.MAPLE(X_train, y_train, X_train, y_train, dstump=dstump)
            fine_tune = time.time() - start

        except:
            if logger:
                logger.info('maple fine-tuning exceeded 24h!')
            global maple_limit_reached
            maple_limit_reached = True
            return None, None

    start = time.time()
    maple.explain(X_test[test_ndx]) if dstump else maple.get_weights(X_test[test_ndx])
    test_time = time.time() - start

    return fine_tune, test_time


def _teknn_method(tree, args, test_ndx, X_train, y_train, X_val, X_test, logger=None):
    """
    TEKNN fine tuning and computation.
    """

    with timeout(seconds=ONE_DAY):
        try:
            start = time.time()
            extractor = trex.TreeExtractor(tree, tree_kernel=args.tree_kernel)
            X_train_alt = extractor.fit_transform(X_train)
            X_val_alt = extractor.transform(X_val)

            knn_clf, params = exp_util.tune_knn(X_train_alt, y_train, tree, X_val, X_val_alt,
                                                logger=logger)
            fine_tune = time.time() - start

            if logger:
                logger.info('n_neighbors: {}'.format(params['n_neighbors']))

        except:
            if logger:
                logger.info('teknn fine-tuning exceeded 24h!')

            global knn_limit_reached
            knn_limit_reached = True
            return None, None

    start = time.time()
    x_test_alt = extractor.transform(X_test[test_ndx])
    distances, neighbor_ids = knn_clf.kneighbors(x_test_alt)
    test_time = time.time() - start

    return fine_tune, test_time


def experiment(args, logger, out_dir, seed):
    """
    Main method that trains a tree ensemble, then compares the
    runtime of different methods to explain a single test instance.
    """

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    random_state=seed)
    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir)
    X_train, X_test, y_train, y_test, label = data
    X_val = exp_util.get_val_data(X_train, args.val_frac, seed)

    logger.info('train instances: {:,}'.format(len(X_train)))
    logger.info('val instances: {:,}'.format(len(X_val)))
    logger.info('test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train,
                           X_test=X_test, y_test=y_test,
                           logger=logger)
    logger.info('')

    # randomly pick test instances to explain
    np.random.seed(seed)
    test_ndx = np.random.choice(len(y_test), size=1, replace=False)

    # train on predicted labels
    train_label = y_train if args.true_label else model.predict(X_train)

    # TREX
    if args.trex:
        logger.info('TREX...')
        fine_tune, test_time = _trex_method(test_ndx, X_test, model, X_train, train_label, args,
                                            seed=seed, logger=logger)

        logger.info('fine tune: {:.3f}s'.format(fine_tune))
        logger.info('computation time: {:.3f}s'.format(test_time))
        r = {'fine_tune': fine_tune, 'test_time': test_time}
        np.save(os.path.join(out_dir, 'trex_{}.npy'.format(args.kernel_model)), r)

    # Leaf Influence
    if args.tree_type == 'cb' and args.inf_k is not None:
        logger.info('leafinfluence...')
        fine_tune, test_time = _influence_method(model, test_ndx, X_train,
                                                 y_train, X_test, y_test, args.inf_k)

        logger.info('fine tune: {:.3f}s'.format(fine_tune))
        logger.info('computation time: {:.3f}s'.format(test_time))
        r = {'fine_tune': fine_tune, 'test_time': test_time}
        np.save(os.path.join(out_dir, 'influence.npy'), r)

    if args.maple and not maple_limit_reached:
        logger.info('maple...')
        fine_tune, test_time = _maple_method(model, test_ndx, X_train, train_label, X_test, y_test,
                                             dstump=args.dstump, logger=logger)

        if fine_tune is not None and test_time is not None:
            logger.info('fine tune: {:.3f}s'.format(fine_tune))
            logger.info('computation time: {:.3f}s'.format(test_time))
            r = {'fine_tune': fine_tune, 'test_time': test_time}
            np.save(os.path.join(out_dir, 'maple.npy'), r)

        else:
            logger.info('time limit reached!')

    if args.teknn and not teknn_limit_reached:
        logger.info('knn...')
        fine_tune, test_time = _teknn_method(model, args, test_ndx, X_train, train_label,
                                             X_val, X_test, logger=logger)
        if fine_tune is not None and test_time is not None:
            logger.info('fine tune: {:.3f}s'.format(fine_tune))
            logger.info('computation time: {:.3f}s'.format(test_time))
            r = {'fine_tune': fine_tune, 'test_time': test_time}
            np.save(os.path.join(out_dir, 'teknn.npy'), r)

        else:
            logger.info('time limit reached!')


def main(args):

    # make logger
    out_dir = os.path.join(args.out_dir, args.dataset, args.tree_type,
                           args.tree_kernel, 'rs{}'.format(args.rs))
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/runtime/', help='output directory.')

    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--val_frac', type=float, default=0.05, help='Amount of data for validation.')

    parser.add_argument('--tree_type', type=str, default='cb', help='Model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth in tree ensemble.')

    parser.add_argument('--trex', action='store_true', default=False, help='TREX method.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='Type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='Kernel model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='Similarity kernel')
    parser.add_argument('--true_label', action='store_true', default=False, help='Train TREX on the true labels.')

    parser.add_argument('--teknn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves for leafinfluence.')
    parser.add_argument('--maple', action='store_true', default=False, help='Run experiment using MAPLE.')
    parser.add_argument('--dstump', action='store_true', default=False, help='Enable DSTUMP with Maple.')

    parser.add_argument('--rs', type=int, default=1, help='Random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)


class Args:
    dataset = 'mfc19_mfc20'
    data_dir = 'data'
    out_dir = 'output/runtime/'

    train_frac = 1.0
    val_frac = 0.1

    tree_type = 'cb'
    n_estimators = 100
    max_depth = None

    trex = True
    tree_kernel = 'leaf_output'
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'
    true_label = False

    teknn = False
    inf_k = None
    maple = False
    dstump = True

    rs = 1
