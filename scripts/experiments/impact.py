"""
Experiment:
    1) Select a test instance uniformly at random.
    2) Sort training instances by absolute difference on the test instance probability OR loss.
    3) Remove 1 or 10 or 100 or etc. most impactful training instances.
    4) Train a new tree-ensemble on the reduced training dataset.
    5) Measure absolute change in predicted probability.

The best methods choose samples to be removed that have the highest change
in absolute predicted probabiltiy on the test sample.
"""
import os
import sys
import time
import uuid
import shutil
import resource
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
from copy import deepcopy
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility
import trex
import util
from baselines.influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from baselines.maple.MAPLE import MAPLE


def score(model, X_test, y_test):
    """
    Evaluates the model the on test set and returns metric scores.
    """

    # 1 test sample
    if y_test.shape[0] == 1:
        result = (-1, -1)

    # >1 test sample
    else:
        acc = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return (acc, auc)

    return result


def measure_performance(train_indices, clf, X_train, y_train, X_test, y_test, logger=None):
    """
    Measures the change in predictions as training instances are removed.
    """

    # baseline predicted probability
    model = clone(clf).fit(X_train, y_train)
    base_proba = model.predict_proba(X_test)[:, 1]

    if logger:
        logger.info('\nremoving, retraining, and remeasuring predicted probability...')
        logger.info('test label: {}, before prob.: {:.5f}'.format(int(y_test[0]), base_proba[0]))

    result = {}
    result['proba_delta'] = []
    result['remove_pct'] = []

    # fraction for removing one sample
    frac_zero = 1 / X_train.shape[0]
    args.train_frac_to_remove.insert(0, frac_zero)

    # compute how many samples should be removed
    for train_frac_to_remove in args.train_frac_to_remove:
        n_remove = round(train_frac_to_remove * X_train.shape[0])
        remove_indices = train_indices[:n_remove]

        # remove most influential training samples
        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        # measure change in test instance predictive probability
        new_model = clone(clf).fit(new_X_train, new_y_train)
        proba = new_model.predict_proba(X_test)[:, 1]
        proba_diff = np.abs(base_proba - proba)[0]

        # display status
        if logger:
            s = '[{:.5f}% removed] after prob.: {:.5f}, delta: {:.5f}'
            logger.info(s.format(train_frac_to_remove * 100, proba[0], proba_diff))

        # add to results
        result['proba_delta'].append(np.abs(base_proba - proba)[0])
        result['remove_pct'].append(train_frac_to_remove * 100)

    return result


def random_method(X_train, rng):
    """
    Randomly orders the training intances to be removed.
    """
    return rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)


def trex_method(args, model, X_train, y_train, X_test, logger=None,
                frac_progress_update=0.1):
    """
    Sort training instances by largest 'excitatory' influnce on the test set.
    """

    # train surrogate model
    kernel_model = args.method.split('-')[0].split('_')[0]
    tree_kernel = args.method.split('-')[-1]
    surrogate = trex.TreeExplainer(model,
                                   X_train,
                                   y_train,
                                   kernel_model=kernel_model,
                                   tree_kernel=tree_kernel,
                                   val_frac=args.tune_frac,
                                   metric=args.metric,
                                   random_state=args.rs,
                                   logger=logger)

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # sort instances with highest positive influence first
    attributions_sum = np.zeros(X_train.shape[0])

    # compute impact of each training sample on the test set
    for i in range(X_test.shape[0]):
        attributions_sum += surrogate.compute_attributions(X_test[[i]])[0]

    # sort instances most inhibitory samples first
    # train_indices = np.argsort(attributions_sum)[::-1]
    train_indices = np.argsort(attributions_sum)
    # train_indices = np.argsort(np.abs(attributions_sum))[::-1]

    # print(attributions_sum[train_indices])

    return train_indices


def maple_method(args, model, X_train, y_train, X_test, logger=None,
                 frac_progress_update=0.1):
    """
    Sort training instances using MAPLE's "local training distribution".
    """

    # train a MAPLE explainer model
    train_label = model.predict(X_train)
    maple_explainer = MAPLE(X_train, train_label, X_train, train_label,
                            verbose=args.verbose, dstump=False)

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test instance...')

    # contributions container
    contributions_sum = np.zeros(X_train.shape[0])

    # compute similarity of each training instance to the set set
    for i in range(X_test.shape[0]):
        contributions = maple_explainer.get_weights(X_test[i])
        contributions_sum += contributions

    # sort training instances based on similarity to the test set
    train_indices = np.argsort(contributions_sum)[::-1]

    return train_indices


def influence_method(args, model, X_train, y_train, X_test, y_test, logger=None,
                     k=-1, frac_progress_update=0.1):
    """
    Sort training instances based on their Leaf Influence on the test set.

    Reference:
    https://github.com/kohpangwei/influence-release/blob/master/influence/experiments.py
    """
    assert k == -1, 'AllPoints method not used for k: {}'.format(k)
    assert args.model == 'cb', 'tree-ensemble is not a CatBoost model!'

    # save CatBoost model
    temp_dir = os.path.join('.catboost_info', 'leaf_influence_{}'.format(str(uuid.uuid4())))
    temp_fp = os.path.join(temp_dir, 'cb.json')
    os.makedirs(temp_dir, exist_ok=True)
    model.save_model(temp_fp, format='json')

    # initialize Leaf Influence
    explainer = CBLeafInfluenceEnsemble(temp_fp, X_train, y_train, k=k,
                                        learning_rate=model.learning_rate_,
                                        update_set='AllPoints')

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test instance...')

    # contributions container
    start = time.time()
    contributions_sum = np.zeros(X_train.shape[0])

    # compute influence on each test instance
    for i in range(X_test.shape[0]):

        contributions = []
        buf = deepcopy(explainer)

        # compute influence for each training instance
        for j in range(X_train.shape[0]):
            explainer.fit(removed_point_idx=j, destination_model=buf)
            contributions.append(buf.loss_derivative(X_test[[i]], y_test[[i]])[0])

            # display progress
            if logger and j % int(X_train.shape[0] * frac_progress_update) == 0:
                elapsed = time.time() - start
                train_frac_complete = j / X_train.shape[0] * 100
                logger.info('train {:.1f}%...{:.3f}s'.format(train_frac_complete, elapsed))

        contributions = np.array(contributions)
        contributions_sum += contributions

    # sort by decreasing absolute influence
    train_indices = np.argsort(np.abs(contributions_sum))[::-1]

    # clean up
    shutil.rmtree(temp_dir)

    return train_indices


def teknn_method(args, model, X_train, y_train, X_test, logger=None,
                 frac_progress_update=0.1):
    """
    Sort trainnig instance based on similarity density to the test instances.
    """

    # transform the data
    tree_kernel = args.method.split('-')[-1]
    extractor = trex.TreeExtractor(model, tree_kernel=tree_kernel)
    X_train_alt = extractor.transform(X_train)

    # train surrogate model
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]}
    surrogate = trex.util.train_surrogate(model,
                                          'knn',
                                          param_grid,
                                          X_train,
                                          X_train_alt,
                                          y_train,
                                          val_frac=args.tune_frac,
                                          metric=args.metric,
                                          seed=args.rs,
                                          logger=logger)

    # display status
    if logger:
        logger.info('\ncomputing influence of all training samples on the test instance...')

    # contributions container
    contributions_sum = np.zeros(X_train.shape[0])

    # compute the contribution of all training samples on each test instance
    for i in range(X_test.shape[0]):
        x_test_alt = extractor.transform(X_test[[i]])
        pred_label = int(surrogate.predict(x_test_alt)[0])
        distances, neighbor_ids = surrogate.kneighbors(x_test_alt)

        # add density to training instances that are in the neighborhood of the test instance
        for neighbor_id in neighbor_ids[0]:
            contribution = 1 if y_train[neighbor_id] == pred_label else -1
            contributions_sum[neighbor_id] += contribution

    # sort instances based on similarity density
    train_indices = np.argsort(contributions_sum)[::-1]

    return train_indices


def sort_train_instances(args, model, X_train, y_train, X_test, y_test, rng, logger=None):
    """
    Sorts training instance to be removed using one of several methods.
    """

    # random method
    if args.method == 'random':
        train_indices = random_method(X_train, rng)

    # TREX method
    elif 'klr' in args.method or 'svm' in args.method:
        train_indices = trex_method(args, model, X_train, y_train, X_test, logger=logger)

    # MAPLE
    elif args.method == 'maple':
        train_indices = maple_method(args, model, X_train, y_train, X_test, logger=logger)

    # Leaf Influence (NOTE: can only compute influence of the LOSS, requires label)
    elif args.method == 'leaf_influence':
        train_indices = influence_method(args, model, X_train, y_train, X_test, y_test, logger=logger)

    # TEKNN
    elif 'knn' in args.method:
        train_indices = teknn_method(args, model, X_train, y_train, X_test, logger=logger)

    else:
        raise ValueError('method {} unknown!'.format(args.method))

    return train_indices


def experiment(args, logger, out_dir):
    """
    Main method that removes training instances ordered by
    different methods and measure their impact on a random
    set of test instances.
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

    # use a fraction of the train data
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train_samples = int(X_train.shape[0] * args.train_frac)
        train_indices = rng.choice(X_train.shape[0], size=n_train_samples, replace=False)
        X_train, y_train = X_train[train_indices], y_train[train_indices]

    # select a subset of test instances uniformly at random
    test_indices = rng.choice(X_test.shape[0], size=args.n_test, replace=False)
    X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

    # display dataset statistics
    logger.info('\nno. train instances: {:,}'.format(X_train.shape[0]))
    logger.info('no. test instances: {:,}'.format(X_test_sub.shape[0]))
    logger.info('no. features: {:,}\n'.format(X_train.shape[1]))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    util.performance(model, X_train, y_train, logger=logger, name='Train')

    # sort train instances, then remove, retrain, and re-evaluate
    train_indices = sort_train_instances(args, model, X_train, y_train, X_test_sub, y_test_sub, rng, logger=logger)
    result = measure_performance(train_indices, clf, X_train, y_train, X_test_sub, y_test_sub, logger=logger)

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['total_time'] = time.time() - begin
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # display results
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))


def main(args):

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
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/impact/', help='directory to save results.')

    # Data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='fraction of train data to evaluate.')
    parser.add_argument('--tune_frac', type=float, default=0.1, help='amount of data for validation.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=10, help='no. of trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='max. depth in tree ensemble.')

    # Method settings
    parser.add_argument('--method', type=str, default='klr-leaf_output', help='method.')
    parser.add_argument('--metric', type=str, default='mse', help='metric for tuning surrogate models.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--train_frac_to_remove', type=float, nargs='+', default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                        help='frac. train instances to remove.')
    parser.add_argument('--n_test', type=int, default=1, help='no. of test instances to evaluate.')

    # Additional settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
