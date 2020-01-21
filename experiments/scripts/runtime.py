"""
Experiment: Compare runtimes for explaining a single test instance for different methods.
"""
import time
import argparse
import os
import sys
import signal
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import numpy as np
from sklearn.base import clone
from maple import MAPLE

import trex
from utility import model_util, data_util, exp_util, print_util

ONE_DAY = 86400  # number of seconds in a day
maple_limit_reached = False
knn_limit_reached = False


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


def _our_method(test_ndx, X_test, model, X_train, y_train, encoding='leaf_output', linear_model='svm', C=1.0,
                kernel='rbf', random_state=69, X_val=None, logger=None):
    """Explains the predictions of each test instance."""

    start = time.time()
    explainer = trex.TreeExplainer(model, X_train, y_train, encoding=encoding, random_state=random_state,
                                   linear_model=linear_model, kernel=kernel, C=C, X_val=X_val, logger=logger,
                                   dense_output=False)
    fine_tune = time.time() - start

    if logger:
        logger.info('C: {}'.format(explainer.C))

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


def _knn_method(tree, encoding, test_ndx, X_train, y_train, X_val, X_test, logger=None):

    with timeout(seconds=ONE_DAY):
        try:
            start = time.time()
            extractor = trex.TreeExtractor(tree, encoding=args.encoding)
            X_train_alt = extractor.fit_transform(X_train)
            X_val_alt = extractor.transform(X_val)
            train_label = y_train if args.true_label else tree.predict(X_train)

            knn_clf, params = exp_util.tune_knn(X_train_alt, train_label, tree, X_val, X_val_alt, logger=logger)
            fine_tune = time.time() - start

            if logger:
                logger.info('n_neighbors: {}, weights: {}'.format(params['n_neighbors'], params['weights']))
        except:
            if logger:
                logger.info('knn fine-tuning exceeded 24h!')
            global knn_limit_reached
            knn_limit_reached = True
            return None, None

    start = time.time()
    x_test_alt = extractor.transform(X_test[test_ndx])
    distances, neighbor_ids = knn_clf.kneighbors(x_test_alt)
    test_time = time.time() - start

    return fine_tune, test_time


def runtime(args):
    """
    Main method that trains a tree ensemble, then compares the runtime of different methods to explain
    a random subset of test instances.
    """

    # write output to logs
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    our_fine_tune, our_test_time = [], []
    inf_fine_tune, inf_test_time = [], []
    maple_fine_tune, maple_test_time = [], []
    knn_fine_tune, knn_test_time = [], []
    seed = args.rs

    for i in range(args.repeats):
        logger.info('\nrun {}, seed: {}'.format(i + 1, seed))

        # get model and data
        clf = model_util.get_classifier(args.model_type, n_estimators=args.n_estimators,
                                        max_depth=args.max_depth, random_state=seed)
        X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=seed,
                                                                     data_dir=args.data_dir)

        # use part of the test data as validation data
        X_val = X_test.copy()
        if args.val_frac < 1.0 and args.val_frac > 0.0:
            X_val = X_val[int(X_val.shape[0] * args.val_frac):]

        logger.info('train instances: {}'.format(len(X_train)))
        logger.info('val instances: {}'.format(len(X_test)))
        logger.info('test instances: {}'.format(len(X_val)))
        logger.info('num features: {}'.format(X_train.shape[1]))

        # train a tree ensemble
        model = clone(clf).fit(X_train, y_train)
        model_util.performance(model, X_test=X_test, y_test=y_test, logger=logger)

        # randomly pick test instances to explain
        np.random.seed(seed)
        test_ndx = np.random.choice(len(y_test), size=1, replace=False)

        # train on predicted labels
        train_label = y_train if args.true_label else model.predict(X_train)

        # our method
        if args.trex:
            logger.info('ours...')
            fine_tune, test_time = _our_method(test_ndx, X_test, model, X_train, train_label,
                                               encoding=args.encoding, C=args.C, linear_model=args.linear_model,
                                               kernel=args.kernel, random_state=seed, X_val=X_val, logger=logger)
            logger.info('fine tune: {:.3f}s'.format(fine_tune))
            logger.info('test time: {:.3f}s'.format(test_time))
            our_fine_tune.append(fine_tune)
            our_test_time.append(test_time)

        # influence method
        if args.model_type == 'cb' and args.inf_k is not None:
            logger.info('leafinfluence...')
            fine_tune, test_time = _influence_method(model, test_ndx, X_train, y_train, X_test, y_test, args.inf_k)
            logger.info('fine tune: {:.3f}s'.format(fine_tune))
            logger.info('test time: {:.3f}s'.format(test_time))
            inf_fine_tune.append(fine_tune)
            inf_test_time.append(test_time)

        if args.maple and not maple_limit_reached:
            logger.info('maple...')
            fine_tune, test_time = _maple_method(model, test_ndx, X_train, train_label, X_test, y_test,
                                                 dstump=args.dstump, logger=logger)
            if fine_tune is not None and test_time is not None:
                logger.info('fine tune: {:.3f}s'.format(fine_tune))
                logger.info('test time: {:.3f}s'.format(test_time))
                maple_fine_tune.append(fine_tune)
                maple_test_time.append(test_time)

        if args.knn and not knn_limit_reached:
            logger.info('knn...')
            fine_tune, test_time = _knn_method(model, args.encoding, test_ndx, X_train, train_label,
                                               X_val, X_test, logger=logger)
            if fine_tune is not None and test_time is not None:
                logger.info('fine tune: {:.3f}s'.format(fine_tune))
                logger.info('test time: {:.3f}s'.format(test_time))
                knn_fine_tune.append(fine_tune)
                knn_test_time.append(test_time)

        seed += 1

    # display results
    if args.trex:
        our_fine_tune = np.array(our_fine_tune)
        our_test_time = np.array(our_test_time)
        logger.info('\nour')
        logger.info('fine tuning: {:.3f}s +/- {:.3f}s'.format(our_fine_tune.mean(), our_fine_tune.std()))
        logger.info('test time: {:.3f}s +/- {:.3f}s'.format(our_test_time.mean(), our_test_time.std()))

    if args.model_type == 'cb' and args.inf_k is not None:
        inf_fine_tune = np.array(inf_fine_tune)
        inf_test_time = np.array(inf_test_time)
        logger.info('\nleafinfluence')
        logger.info('fine tuning: {:.3f}s +/- {:.3f}s'.format(inf_fine_tune.mean(), inf_fine_tune.std()))
        logger.info('test time: {:.3f}s +/- {:.3f}s'.format(inf_test_time.mean(), inf_test_time.std()))

    if args.maple and not maple_limit_reached:
        maple_fine_tune = np.array(maple_fine_tune)
        maple_test_time = np.array(maple_test_time)
        logger.info('\nmaple')
        logger.info('fine tuning: {:.3f}s +/- {:.3f}s'.format(maple_fine_tune.mean(), maple_fine_tune.std()))
        logger.info('test time: {:.3f}s +/- {:.3f}s'.format(maple_test_time.mean(), maple_test_time.std()))

    if args.knn and not knn_limit_reached:
        knn_fine_tune = np.array(knn_fine_tune)
        knn_test_time = np.array(knn_test_time)
        logger.info('\nknn')
        logger.info('fine tuning: {:.3f}s +/- {:.3f}s'.format(knn_fine_tune.mean(), knn_fine_tune.std()))
        logger.info('test time: {:.3f}s +/- {:.3f}s'.format(knn_test_time.mean(), knn_test_time.std()))

    # save results
    if args.save_results:
        exp_dir = os.path.join(args.out_dir, args.dataset)
        os.makedirs(exp_dir, exist_ok=True)

        # ours
        if args.trex:
            setting = '{}_{}'.format(args.linear_model, args.encoding)
        np.save(os.path.join(exp_dir, 'ours_{}_fine_tune.npy'.format(setting)), our_fine_tune)
        np.save(os.path.join(exp_dir, 'ours_{}_test_time.npy'.format(setting)), our_test_time)

        # leafinfluence
        if args.model_type == 'cb' and args.inf_k is not None:
            np.save(os.path.join(exp_dir, 'influence_fine_tune.npy'), inf_fine_tune)
            np.save(os.path.join(exp_dir, 'influence_test_time.npy'), inf_test_time)

        # MAPLE
        if args.maple:
            maple_settings = '' if not args.dstump else 'dstump_'
            np.save(os.path.join(exp_dir, 'maple_{}fine_tune.npy'.format(maple_settings)), maple_fine_tune)
            np.save(os.path.join(exp_dir, 'maple_{}test_time.npy'.format(maple_settings)), maple_test_time)

        # KNN
        if args.knn:
            np.save(os.path.join(exp_dir, 'teknn_fine_tune.npy'), knn_fine_tune)
            np.save(os.path.join(exp_dir, 'teknn_test_time.npy'), knn_test_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/runtime', help='output directory.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')
    parser.add_argument('--model_type', type=str, default='cb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='Similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=1, help='for reproducibility.')
    parser.add_argument('--inf_k', default=None, type=int, help='Number of leaves for leafinfluence.')
    parser.add_argument('--maple', action='store_true', help='Run experiment using MAPLE.')
    parser.add_argument('--dstump', action='store_true', help='Enable DSTUMP with Maple.')
    parser.add_argument('--start_pct', default=100, type=int, help='Percentage of training data to start with.')
    parser.add_argument('--repeats', default=5, type=int, help='Number of times to repeat the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--true_label', action='store_true', help='Train explainers on true labels.')
    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--trex', action='store_true', default=False, help='TREX method.')
    args = parser.parse_args()
    runtime(args)
