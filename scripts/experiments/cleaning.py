"""
Experiment: dataset cleaning from noisy labels,
            specifically a percentage of flipped labels
            in the train data. Only for binary classification datasets.
"""
import os
import sys
import time
import uuid
import json
import resource
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
from copy import deepcopy
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility
import trex
from utility import model_util
from utility import data_util
from utility import exp_util
from utility import print_util
from baselines.influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from baselines.maple.MAPLE import MAPLE
from baselines.mmd_critic import mmd


def score(model, X_test, y_test):
    """
    Evaluates the model the on test set and returns metric scores.
    """
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return acc, auc


def flip_labels(arr, k=100, seed=1, indices=None, logger=None):
    """
    Flips the label of random elements in an array; only for binary arrays.

    If `indices` is None, flip labels of `k` instances at random,
    otherwise flip the labels of specified `indices`.
    """

    # check to make sure `arr` is comprised only of binary labels
    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'

    # select indices to flip uniformly at random
    if indices is None:

        # `k` represent a fraction of instances
        if k <= 1.0:
            assert isinstance(k, float), 'k is not a float!'
            assert k > 0, 'k is less than zero!'
            k = int(len(arr) * k)
        assert k <= len(arr), 'k is greater than len(arr)!'

        # randomly select `k` instances
        rng = np.random.default_rng(1)
        indices = rng.choice(arr.shape[0], size=k, replace=False)

    # new arry with flipped labels
    new_arr = arr.copy()

    # record no. pos. labels flipped
    num_ones_flipped = 0

    # flip label of each instance and record if a pos. label is flipped
    for ndx in indices:
        num_ones_flipped += 1 if new_arr[ndx] == 1 else 0
        new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1

    # make sure changes from old label array match the new array statistics
    num_zeros_flipped = indices.shape[0] - num_ones_flipped
    assert np.sum(new_arr) == np.sum(arr) - num_ones_flipped + num_zeros_flipped

    # display changes
    if logger:
        logger.info('\nsum before: {:,}'.format(np.sum(arr)))
        logger.info('ones flipped: {:,}'.format(num_ones_flipped))
        logger.info('zeros flipped: {:,}'.format(num_zeros_flipped))
        logger.info('sum after: {:,}'.format(np.sum(new_arr)))

    return new_arr, indices


# def record_fixes(train_indices, noisy_indices, n_checkpoint):
def fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                        clf, X_train, y_train, X_test, y_test,
                        acc_noisy, auc_noisy, logger=None):
    """
    Returns the number of train instances checked and which train instances were
    fixed for each checkpoint.
    """

    # initialize results
    result = {}
    result['accs'] = [acc_noisy]
    result['aucs'] = [auc_noisy]
    result['checked_pcts'] = [0]
    result['fixed_pcts'] = [0]

    # result containers
    indices_to_fix = []
    start = time.time()
    s = '[Checkpoint] checked: {:.1f}%, fixed: {:.1f}%, Acc.: {:.3f}, AUC: {:.3f}, cum. time: {:.3f}s'

    # display status
    if logger:
        logger.info('\nchecking and fixing training instances...')

    # incrementers
    n_checked = 0

    # check training instances in the order given by `train_indices`
    for train_ndx in train_indices[:n_check]:

        # flag instance to be fixed if it was flipped
        if train_ndx in noisy_indices:
            indices_to_fix.append(train_ndx)

        # increment no. training samples checked
        n_checked += 1

        # check point reached, fix noisy instances found so far and retrain and re-evaluate
        if n_checked % n_checkpoint == 0:

            # fix indices up to this checkpoint
            semi_noisy_indices = np.setdiff1d(noisy_indices, indices_to_fix)
            y_train_semi_noisy, _ = flip_labels(y_train, indices=semi_noisy_indices)

            # train and evaluate the tree-ensemble model on this less noisy dataset
            model_semi_noisy = clone(clf).fit(X_train, y_train_semi_noisy)
            acc_semi_noisy, auc_semi_noisy = score(model_semi_noisy, X_test, y_test)

            # add to list of results
            result['accs'].append(acc_semi_noisy)
            result['aucs'].append(auc_semi_noisy)
            result['checked_pcts'].append(float(n_checked / y_train.shape[0]) * 100)
            result['fixed_pcts'].append(float(len(indices_to_fix) / y_train.shape[0]) * 100)

            # display progress
            if logger:
                logger.info(s.format(result['checked_pcts'][-1], result['fixed_pcts'][-1],
                                     result['accs'][-1], result['aucs'][-1], time.time() - start))

    return result


def random_method(noisy_indices, n_check, n_checkpoint,
                  clf, X_train, y_train, X_test, y_test,
                  acc_noisy, auc_noisy, seed=1, logger=None):
    """
    Selects train instances to check uniformly at random.
    """
    rng = np.random.default_rng(seed + 1)  # +1 to avoid choosing the same indices as the noisy labels
    train_indices = rng.choice(y_train.shape[0], size=n_check, replace=False)
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)
    return result


def trex_method(args, model_noisy, y_train_noisy,
                noisy_indices, n_check, n_checkpoint,
                clf, X_train, y_train, X_test, y_test,
                acc_noisy, auc_noisy, logger=None):
    """
    Order by largest absolute values of the instance coefficients
    from the KLR or SVM surrogate model.
    """

    # train surrogate model
    kernel_model = args.method.split('-')[0].split('_')[0]
    tree_kernel = args.method.split('-')[-1]
    surrogate = trex.TreeExplainer(model_noisy,
                                   X_train,
                                   y_train_noisy,
                                   kernel_model=kernel_model,
                                   tree_kernel=tree_kernel,
                                   val_frac=args.tune_frac,
                                   metric=args.metric,
                                   random_state=args.rs,
                                   logger=logger)

    # sort by instance log loss using the surrogate model
    if 'loss' in args.method:
        y_train_proba = surrogate.predict_proba(X_train)[:, 1]
        y_train_loss = exp_util.instance_log_loss(y_train, y_train_proba)  # negative log-likelihood
        train_indices = np.argsort(y_train_loss)  # ascending, largest log loss first

    # sort by absolute value of instance weights
    else:
        train_indices = np.argsort(np.abs(surrogate.alpha_))[::-1]

    # fix noisy instances
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)
    return result


def tree_loss_method(model_noisy,
                     noisy_indices, n_check, n_checkpoint,
                     clf, X_train, y_train, X_test, y_test,
                     acc_noisy, auc_noisy, seed=1, logger=None):
    """
    Orders training instances by largest loss.
    """
    y_train_proba = model_noisy.predict_proba(X_train)[:, 1]
    y_train_loss = exp_util.instance_log_loss(y_train, y_train_proba)  # negative log-likelihood
    train_indices = np.argsort(y_train_loss)  # ascending, largest log loss first
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)
    return result


# def leaf_influence_method(explainer, noisy_ndx, X_train, y_train, y_train_noisy, interval, to_check=1):
def leaf_influence_method(model_noisy, y_train_noisy,
                          noisy_indices, n_check, n_checkpoint,
                          clf, X_train, y_train, X_test, y_test,
                          acc_noisy, auc_noisy, seed=1, logger=None,
                          k=-1, out_dir='.', frac_progress_update=0.1):
    """
    Computes the influence on train instance i if train
    instance i were upweighted/removed. This uses the FastLeafInfluence
    method by Sharchilev et al.

    Reference:
    https://github.com/kohpangwei/influence-release/blob/master/influence/experiments.py
    """
    assert k == -1, 'AllPoints method not used for k: {}'.format(k)
    assert args.model == 'cb', 'tree-ensemble is not a CatBoost model!'

    # save CatBoost model
    temp_fp = '.{}_cb.json'.format(str(uuid.uuid4()))
    model_noisy.save_model(temp_fp, format='json')

    # initialize explainer
    explainer = CBLeafInfluenceEnsemble(temp_fp,
                                        X_train,
                                        y_train_noisy,
                                        k=k,
                                        learning_rate=model_noisy.learning_rate_,
                                        update_set='AllPoints')

    # display progress
    if logger:
        logger.info('\ncomputing self-influence of training instances...')
        start = time.time()

    # score container
    influence_scores = []

    # compute self-influence score for each training instance
    buf = deepcopy(explainer)
    for i in range(X_train.shape[0]):
        explainer.fit(removed_point_idx=i, destination_model=buf)
        influence_scores.append(buf.loss_derivative(X_train[[i]], y_train_noisy[[i]])[0])

        # display progress
        if logger and i % int(X_train.shape[0] * frac_progress_update) == 0:
            elapsed = time.time() - start
            logger.info('finished {:.1f}% train instances...{:.3f}s'.format((i / X_train.shape[0]) * 100, elapsed))

    # convert scores to a numpy array
    influence_scores = np.array(influence_scores)

    # sort by ascending order; the most negative train instances
    # are the ones that increase the log loss the most, and are the most harmful
    train_indices = np.argsort(influence_scores)
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)

    # clean up
    os.system('rm {}'.format(temp_fp))

    return result


# def maple_method(explainer, X_train, noisy_ndx, interval, to_check=1):
def maple_method(model_noisy,
                 noisy_indices, n_check, n_checkpoint,
                 clf, X_train, y_train, X_test, y_test,
                 acc_noisy, auc_noisy, seed=1, logger=None,
                 frac_progress_update=0.1):
    """
    Orders instances by tree kernel similarity density.
    """

    # display progress
    if logger:
        logger.info('\ncomputing similarity density...')
        start = time.time()

    # train explainer
    train_label = model_noisy.predict(X_train)
    explainer = MAPLE(X_train, train_label, X_train, train_label,
                      verbose=args.verbose, dstump=False)

    # compute similarity density per training instance
    train_weight = np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):
        weights = explainer.get_weights(X_train[i])
        train_weight[i] += np.sum(weights)

        # display progress
        if logger and i % int(X_train.shape[0] * frac_progress_update) == 0:
            elapsed = time.time() - start
            logger.info('finished {:.1f}% train instances...{:.3f}s'.format((i / X_train.shape[0]) * 100, elapsed))

    # sort by training instance densities
    train_indices = np.argsort(train_weight)[::-1]
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)
    return result


# def knn_method(knn_clf, X_train, noisy_ndx, interval, to_check=1):
def teknn_method(model_noisy, y_train_noisy,
                 noisy_indices, n_check, n_checkpoint,
                 clf, X_train, y_train, X_test, y_test,
                 acc_noisy, auc_noisy, seed=1, logger=None,
                 frac_progress_update=0.1):
    """
    Count impact by the number of times a training sample shows up in
    one another's neighborhood, this can be weighted by 1 / distance.
    """

    # transform the data
    tree_kernel = args.method.split('-')[-1]
    extractor = trex.TreeExtractor(model_noisy, tree_kernel=tree_kernel)
    X_train_alt = extractor.fit_transform(X_train)

    # train surrogate model
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]}
    surrogate = trex.util.train_surrogate(model_noisy,
                                          'knn',
                                          param_grid,
                                          X_train,
                                          X_train_alt,
                                          y_train_noisy,
                                          val_frac=args.tune_frac,
                                          metric=args.metric,
                                          seed=args.rs,
                                          logger=logger)

    # sort by instance log loss using the surrogate model
    if 'loss' in args.method:

        # display progress
        if logger:
            logger.info('\ncomputing KNN loss...')

        y_train_proba = surrogate.predict_proba(X_train_alt)[:, 1]
        y_train_loss = exp_util.instance_log_loss(y_train, y_train_proba)  # negative log-likelihood
        train_indices = np.argsort(y_train_loss)  # ascending, largest log loss first

    # sort by absolute value of instance weights
    else:

        # display progress
        if logger:
            logger.info('\ncomputing similarity density...')
            start = time.time()

        # compute similarity density
        train_weight = np.zeros(X_train_alt.shape[0])
        for i in range(X_train_alt.shape[0]):
            distances, neighbor_ids = surrogate.kneighbors([X_train_alt[i]])
            train_weight[neighbor_ids[0]] += 1

            # display progress
            if logger and i % int(X_train.shape[0] * frac_progress_update) == 0:
                elapsed = time.time() - start
                logger.info('finished {:.1f}% train instances...{:.3f}s'.format((i / X_train.shape[0]) * 100, elapsed))

        # sort instances by their similarity density
        train_indices = np.argsort(train_weight)[::-1]

    # fix noisy instances
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)
    return result


# def proto_method(model, X_train, y_train, noisy_ndx, interval, n_check):
def tree_prototype_method(model_noisy, y_train_noisy,
                          noisy_indices, n_check, n_checkpoint,
                          clf, X_train, y_train, X_test, y_test,
                          acc_noisy, auc_noisy, seed=1, logger=None,
                          k=10, frac_progress_update=0.1):
    """
    Orders instances by using the GBT distance similarity formula.
    It then ranks training samples based on the proportion of
    labels from the k = 10 nearest neighbors.

    Reference:
    https://arxiv.org/pdf/1611.07115.pdf.
    """

    # get feature extractor
    extractor = trex.TreeExtractor(model_noisy, tree_kernel='leaf_path')
    X_train_alt = extractor.fit_transform(X_train)

    # obtain weight of each tree: note, this code is specific to CatBoost
    temp_fp = '.{}_cb.json'.format(str(uuid.uuid4()))
    model_noisy.save_model(temp_fp, format='json')
    cb_dump = json.load(open(temp_fp, 'r'))

    # obtain weight of each tree: learning_rate^2 * var(predictions)
    tree_weights = []
    for tree in cb_dump['oblivious_trees']:
        predictions = []

        for val, weight in zip(tree['leaf_values'], tree['leaf_weights']):
            predictions += [val] * weight

        tree_weights.append(np.var(predictions) * (model_noisy.learning_rate_ ** 2))

    # weight leaf path feature representation by the tree weights
    for i in range(X_train_alt.shape[0]):

        weight_cnt = 0
        for j in range(X_train_alt.shape[1]):

            if X_train_alt[i][j] == 1:
                X_train_alt[i][j] *= tree_weights[weight_cnt]
                weight_cnt += 1

        assert weight_cnt == len(tree_weights)

    # build a KNN using this proximity measure using k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = knn.fit(X_train_alt, y_train_noisy)

    # compute proportion of neighbors that share the same label
    start = time.time()
    train_weight = np.zeros(X_train_alt.shape[0])
    for i in range(X_train_alt.shape[0]):
        _, neighbor_ids = knn.kneighbors([X_train_alt[i]])
        train_weight[i] = len(np.where(y_train_noisy[i] == y_train_noisy[neighbor_ids[0]])[0]) / len(neighbor_ids[0])

        # display progress
        if logger and i % int(X_train.shape[0] * frac_progress_update) == 0:
            elapsed = time.time() - start
            logger.info('finished {:.1f}% train instances...{:.3f}s'.format((i / X_train.shape[0]) * 100, elapsed))

    # rank training instances by low label agreement with its neighbors
    train_indices = np.argsort(train_weight)
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)

    # clean up
    os.system('rm {}'.format(temp_fp))

    return result


# def mmd_protoype_method(model, X_train, y_train, noisy_ndx, interval, n_check):
def mmd_prototype_method(model_noisy, y_train_noisy,
                         noisy_indices, n_check, n_checkpoint,
                         clf, X_train, y_train, X_test, y_test,
                         acc_noisy, auc_noisy, seed=1, logger=None, n_prototypes=10):
    """
    Orders instances by prototypes and/or criticisms.
    """

    # initialize
    n_criticisms = n_check
    X = np.hstack([X_train, y_train_noisy.reshape(-1, 1)])
    K = rbf_kernel(X)
    candidates = np.arange(len(K))

    # display progress
    if logger:
        logger.info('\ncomputing MMD prototypes...')

    # select prototypes and criticism instances
    prototypes = mmd.greedy_select_protos(K, candidates, n_prototypes)
    criticisms = mmd.select_criticism_regularized(K, prototypes, n_criticisms, is_K_sparse=False)

    # sort indices by criticisms and fix instances
    train_indices = criticisms
    result = fix_noisy_instances(train_indices, noisy_indices, n_check, n_checkpoint,
                                 clf, X_train, y_train, X_test, y_test,
                                 acc_noisy, auc_noisy, logger=logger)
    return result


def experiment(args, logger, out_dir):
    """
    Cleaning Experiment:
      1) Train a tree ensemble.
      2) Flip a percentage of train labels.
      3) Prioritize train instances to be checked using various methods.
      4) Check and correct any flipped train labels.
      5) Compute how effective each method is at cleaning the data.
    """

    # start timer
    begin = time.time()

    # get data
    data = data_util.get_data(args.dataset,
                              data_dir=args.data_dir,
                              processing_dir=args.processing_dir)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # get tree-ensemble
    clf = model_util.get_model(args.model,
                               n_estimators=args.n_estimators,
                               max_depth=args.max_depth,
                               random_state=args.rs,
                               cat_indices=cat_indices)

    # use a subset of the training data
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train = int(X_train.shape[0] * args.train_frac)
        X_train, y_train = X_train[:n_train], y_train[:n_train]

    logger.info('\nno. train instances: {:,}'.format(len(X_train)))
    logger.info('no. test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # add noise
    y_train_noisy, noisy_indices = flip_labels(y_train, k=args.flip_frac, seed=args.rs, logger=logger)
    noisy_indices = np.array(sorted(noisy_indices))
    logger.info('no. noisy labels: {:,}'.format(noisy_indices.shape[0]))

    # train a tree ensemble on the clean and noisy labels
    model = clone(clf).fit(X_train, y_train)
    model_noisy = clone(clf).fit(X_train, y_train_noisy)

    # show model performance before and after noise
    logger.info('\nBefore noise:')
    model_util.performance(model, X_train, y_train, logger=logger, name='Before, Train')
    model_util.performance(model, X_test, y_test, logger=logger, name='Before, Test')

    logger.info('\nAfter noise:')
    model_util.performance(model_noisy, X_train, y_train_noisy, logger=logger, name='After, Noisy Train')
    model_util.performance(model_noisy, X_train, y_train, logger=logger, name='After, Clean Train')
    model_util.performance(model_noisy, X_test, y_test, logger=logger, name='After, Test')

    # check predictive performance before and after noise
    acc_clean, auc_clean = score(model, X_test, y_test)
    acc_noisy, auc_noisy = score(model_noisy, X_test, y_test)

    # find how many corrupted / non-corrupted labels were incorrectly predicted
    predicted_labels = model_noisy.predict(X_train).flatten()
    incorrect_indices = np.where(y_train_noisy != predicted_labels)[0]
    incorrect_noisy_indices = np.intersect1d(noisy_indices, incorrect_indices)
    logger.info('\nno. incorrectly predicted noisy train instances: {:,}'.format(incorrect_noisy_indices.shape[0]))
    logger.info('no. incorrectly predicted train instances: {:,}'.format(incorrect_indices.shape[0]))

    # total no. instances to check and no. instances to check between checkpoints
    n_check = int(y_train.shape[0] * args.check_pct)
    n_checkpoint = int(n_check / args.n_checkpoints)

    # random
    if args.method == 'random':
        result = random_method(noisy_indices, n_check, n_checkpoint,
                               clf, X_train, y_train, X_test, y_test,
                               acc_noisy, auc_noisy, seed=args.rs, logger=logger)

    # TREX
    elif 'klr' in args.method or 'svm' in args.method:
        result = trex_method(args, model_noisy, y_train_noisy,
                             noisy_indices, n_check, n_checkpoint,
                             clf, X_train, y_train, X_test, y_test,
                             acc_noisy, auc_noisy, logger=logger)

    # tree-esemble loss
    elif args.method == 'tree_loss':
        result = tree_loss_method(model_noisy,
                                  noisy_indices, n_check, n_checkpoint,
                                  clf, X_train, y_train, X_test, y_test,
                                  acc_noisy, auc_noisy, seed=args.rs, logger=logger)

    # Leaf Influence
    elif args.method == 'leaf_influence':
        result = leaf_influence_method(model_noisy, y_train_noisy,
                                       noisy_indices, n_check, n_checkpoint,
                                       clf, X_train, y_train, X_test, y_test,
                                       acc_noisy, auc_noisy, seed=args.rs, logger=logger)

    # MAPLE
    elif args.method == 'maple':
        result = maple_method(model_noisy,
                              noisy_indices, n_check, n_checkpoint,
                              clf, X_train, y_train, X_test, y_test,
                              acc_noisy, auc_noisy, seed=args.rs, logger=logger)

    # TEKNN
    elif 'knn' in args.method:
        result = teknn_method(model_noisy, y_train_noisy,
                              noisy_indices, n_check, n_checkpoint,
                              clf, X_train, y_train, X_test, y_test,
                              acc_noisy, auc_noisy, seed=args.rs, logger=logger)

    # Tree Prototype
    elif args.method == 'tree_prototype':
        result = tree_prototype_method(model_noisy, y_train_noisy,
                                       noisy_indices, n_check, n_checkpoint,
                                       clf, X_train, y_train, X_test, y_test,
                                       acc_noisy, auc_noisy, seed=args.rs, logger=logger)

    # MMD Prototype
    elif args.method == 'mmd_prototype':
        result = mmd_prototype_method(model_noisy, y_train_noisy,
                                      noisy_indices, n_check, n_checkpoint,
                                      clf, X_train, y_train, X_test, y_test,
                                      acc_noisy, auc_noisy, seed=args.rs, logger=logger)

    else:
        raise ValueError('unknown method {}'.format(args.method))

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['total_time'] = time.time() - begin
    result['acc_clean'] = acc_clean
    result['auc_clean'] = auc_clean
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # display results
    logger.info('\nResults:\n{}'.format(result))


def main(args):

    # change name of dataset if using a subset of the dataset
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        dataset = '{}_{}'.format(args.dataset, str(args.train_frac).replace('.', 'p'))

    # leave name unchanged
    else:
        dataset = args.dataset

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           dataset,
                           args.model,
                           args.method,
                           'rs_{}'.format(args.rs))

    # create output directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    print_util.clear_dir(out_dir)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--processing_dir', type=str, default='standard', help='processing type.')
    parser.add_argument('--out_dir', type=str, default='output/cleaning/', help='output directory.')

    # Data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='amount of training data to use.')
    parser.add_argument('--tune_frac', type=float, default=0.1, help='amount of training data to use for validation.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth in tree ensemble.')

    # Method settings
    parser.add_argument('--method', type=str, default='klr-leaf_output', help='explanation method.')
    parser.add_argument('--metric', type=str, default='mse', help='fidelity metric to use for TREX.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--flip_frac', type=float, default=0.4, help='fraction of train labels to flip.')
    parser.add_argument('--check_pct', type=float, default=0.3, help='max percentage of train instances to check.')
    parser.add_argument('--n_checkpoints', type=int, default=10, help='number of points to plot.')

    # Additional Settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
