"""
Experiment:
    1) Select a test instance uniformly at random.
    2) Sort training instances by absolute difference on the test instance probability OR loss.
    3) Training samples in equal increments.
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
from scipy.stats import mode

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility
import trex
import util
from baselines.influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from baselines.maple.MAPLE import MAPLE


def measure_performance(args, train_indices, clf, X_train, y_train, X_test, y_test,
                        logger=None):
    """
    Measures the change in predictions as training instances are removed.
    """

    # baseline predicted probability
    model = clone(clf).fit(X_train, y_train)
    base_proba = model.predict_proba(X_test)[:, 1]

    # display status
    if logger:
        logger.info('\nremoving, retraining, and remeasuring predicted probability...')
        logger.info('test label: {}, before prob.: {:.5f}'.format(int(y_test[0]), base_proba[0]))

    # result container
    result = {}
    result['proba_diff'] = [0]
    result['remove_pct'] = [0]

    # compute no. samples to remove between each checkpoint
    n_checkpoint = int(X_train.shape[0] * args.train_frac_to_remove / args.n_checkpoints)

    # remove percentages of training samples and retrain
    for i in range(args.n_checkpoints):
        start = time.time()

        # compute how many samples should be removed
        n_remove = (i + 1) * n_checkpoint
        remove_indices = train_indices[:n_remove]
        pct_remove = n_remove / X_train.shape[0] * 100

        # remove most influential training samples
        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        # only samples from one class remain
        if len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        # measure change in test instance probability
        new_model = clone(clf).fit(new_X_train, new_y_train)
        proba = new_model.predict_proba(X_test)[:, 1]
        proba_diff = np.abs(base_proba - proba)[0]

        # add to results
        result['proba_diff'].append(np.abs(base_proba - proba)[0])
        result['remove_pct'].append(pct_remove)

        # display progress
        if logger:
            s = '[{:.1f}% removed] after prob.: {:.5f}, delta: {:.5f}...{:.3f}s'
            logger.info(s.format(n_remove / X_train.shape[0] * 100, proba[0], proba_diff, time.time() - start))

    return result


def random_method(args, model, X_train, y_train, X_test, rng):
    """
    Randomly orders the training intances to be removed.
    """

    # remove training instances at random
    if args.method == 'random':
        result = rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)

    # remove ONLY minority class training instances
    elif args.method == 'random_minority':
        majority_label = mode(y_train).mode[0]
        minority_label = 1 if majority_label == 0 else 0
        minority_indices = np.where(y_train == minority_label)[0]
        result = rng.choice(minority_indices, size=minority_indices.shape[0], replace=False)

    # remove ONLY majority class training instances
    elif args.method == 'random_majority':
        majority_label = mode(y_train).mode[0]
        majority_indices = np.where(y_train == majority_label)[0]
        result = rng.choice(majority_indices, size=majority_indices.shape[0], replace=False)

    # removes samples ONLY from the predicted label class
    elif args.method == 'random_pred':
        model_pred = model.predict(X_test)[0]
        pred_indices = np.where(y_train == model_pred)[0]
        result = rng.choice(pred_indices, size=pred_indices.shape[0], replace=False)

    else:
        raise ValueError('unknown method {}'.format(args.method))

    return result


def trex_method(args, model, X_train, y_train, X_test, logger=None,
                frac_progress_update=0.1):
    """
    Sort training instances by largest 'excitatory' influnce on the test set.

    There is a difference between
    1) the trainnig instances with the largest excitatory (or inhibitory) attributions, and
    2) the training instances with the largest influence on the PREDICTED labels
       (i.e. excitatory or inhibitory attributions w.r.t. the predicted label).
    """

    # train surrogate model
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

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # sort instances with the larget influence on the predicted labels of the test set
    pred = model.predict(X_test)
    attributions = surrogate.pred_influence(X_test, pred)
    attributions_sum = np.sum(attributions, axis=0)
    train_indices = np.argsort(attributions_sum)[::-1]

    # display attributions from the top k train instances
    if logger:
        k = 20
        sim_s = surrogate.similarity(X_test)[0][train_indices]
        alpha_s = surrogate.get_alpha()[train_indices]
        attributions_sum_s = attributions_sum[train_indices]
        y_train_s = y_train[train_indices]

        train_info = list(zip(train_indices, attributions_sum_s, y_train_s, alpha_s, sim_s))
        s = '[{:5}] label: {}, alpha: {:.3f}, sim: {:.3f} attribution sum: {:.3f}'

        for ndx, atr, lab, alpha, sim in train_info[:k]:
            logger.info(s.format(ndx, lab, alpha, sim, atr))

    return train_indices


def maple_method(args, model, X_train, y_train, X_test, logger=None,
                 frac_progress_update=0.1):
    """
    Sort training instances using MAPLE's "local training distribution".
    """

    # train a MAPLE explainer model
    train_label = model.predict(X_train)
    pred_label = model.predict(X_test)
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

        # consider train instances with the same label as the predicted label as excitatory, else inhibitory
        if '+' in args.method:
            contributions = np.where(train_label == pred_label[i], contributions, contributions * -1)

        contributions_sum += contributions

    # sort train data based largest similarity (MAPLE) OR largest similarity-influence (MAPLE+) to test set
    train_indices = np.argsort(contributions_sum)[::-1]

    return train_indices


def influence_method(args, model, X_train, y_train, X_test, y_test, logger=None,
                     k=-1, update_set='AllPoints', frac_progress_update=0.1):
    """
    Sort training instances based on their Leaf Influence on the test set.

    Reference:
    https://github.com/kohpangwei/influence-release/blob/master/influence/experiments.py
    """

    # LeafInfluence settings
    if 'fast' in args.method:
        k = 0
        update_set = 'SinglePoint'

    assert args.model == 'cb', 'tree-ensemble is not a CatBoost model!'

    # save CatBoost model
    temp_dir = os.path.join('.catboost_info', 'leaf_influence_{}'.format(str(uuid.uuid4())))
    temp_fp = os.path.join(temp_dir, 'cb.json')
    os.makedirs(temp_dir, exist_ok=True)
    model.save_model(temp_fp, format='json')

    # initialize Leaf Influence
    explainer = CBLeafInfluenceEnsemble(temp_fp,
                                        X_train,
                                        y_train,
                                        k=k,
                                        learning_rate=model.learning_rate_,
                                        update_set=update_set)

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

    # train surrogate model
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

    # sort instances based on largest influence on predicted test labels
    attributions = surrogate.compute_attributions(X_test)
    attributions_sum = np.sum(attributions, axis=0)
    train_indices = np.argsort(attributions_sum)[::-1]

    return train_indices


def sort_train_instances(args, model, X_train, y_train, X_test, y_test, rng, logger=None):
    """
    Sorts training instance to be removed using one of several methods.
    """

    # random
    if 'random' in args.method:
        train_indices = random_method(args, model, X_train, y_train, X_test, rng)

    # TREX method
    elif 'klr' in args.method or 'svm' in args.method:
        train_indices = trex_method(args, model, X_train, y_train, X_test, logger=logger)

    # MAPLE
    elif 'maple' in args.method:
        train_indices = maple_method(args, model, X_train, y_train, X_test, logger=logger)

    # Leaf Influence (NOTE: can only compute influence of the LOSS, requires label)
    elif 'leaf_influence' in args.method and args.model == 'cb':
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
    result = measure_performance(args, train_indices, clf, X_train, y_train, X_test_sub, y_test_sub, logger=logger)

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['total_time'] = time.time() - begin
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # display results
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))


def main(args):

    # define output directory
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
    parser.add_argument('--tune_frac', type=float, default=0.0, help='amount of data for validation.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=10, help='no. of trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='max. depth in tree ensemble.')

    # Method settings
    parser.add_argument('--method', type=str, default='klr', help='method.')
    parser.add_argument('--metric', type=str, default='mse', help='metric for tuning surrogate models.')

    # No tuning settings
    parser.add_argument('--C', type=float, default=1.0, help='penalty parameters for KLR or SVM.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='no. neighbors to use for KNN.')
    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='tree kernel.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--n_test', type=int, default=1, help='no. of test instances to evaluate.')
    parser.add_argument('--train_frac_to_remove', type=float, default=0.5, help='fraction of train data to remove.')
    parser.add_argument('--n_checkpoints', type=int, default=10, help='no. checkpoints to perform retraining.')

    # Additional settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
