"""
Experiment: dataset cleaning from noisy labels,
            specifically a percentage of flipped labels
            in the train data. Only for binary classification datasets.
"""
import time
import uuid
import argparse
from datetime import datetime
from copy import deepcopy
import os
import sys
import json
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import tqdm
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier

import trex
from utility import model_util, data_util, exp_util, print_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from maple.MAPLE import MAPLE
from mmd_critic import mmd

TREE_PROTO_K = 10
N_PROTOTYPES = 10


def _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy):
    """
    Retrains the tree ensemble for each ckeckpoint, where a checkpoint represents
    which flipped labels have been fixed.
    """

    X_train, y_train, X_test, y_test = data

    accs = [acc_test_noisy]
    checked_pct = [0]
    fix_pct = [0]

    for n_checked, n_fix in tqdm.tqdm(ckpt_ndx):
        fix_indices = fix_ndx[:n_fix]

        semi_noisy_ndx = np.setdiff1d(noisy_ndx, fix_indices)
        y_train_semi_noisy = data_util.flip_labels_with_indices(y_train, semi_noisy_ndx)

        model_semi_noisy = clone(clf).fit(X_train, y_train_semi_noisy)
        acc_test = accuracy_score(y_test, model_semi_noisy.predict(X_test))

        accs.append(acc_test)
        checked_pct.append(float(n_checked / len(y_train)))
        fix_pct.append(float(n_fix / len(noisy_ndx)))

    return checked_pct, accs


def _record_fixes(train_order, noisy_ndx, train_len, interval):
    """
    Returns the number of train instances checked and which train instances were
    fixed for each checkpoint.
    """

    fix_ndx = []
    ckpt_ndx = []
    checked = 0
    snapshot = 1

    for train_ndx in train_order:
        if train_ndx in noisy_ndx:
            fix_ndx.append(train_ndx)
        checked += 1

        if float(checked / train_len) >= (snapshot * interval):
            ckpt_ndx.append((checked, len(fix_ndx)))
            snapshot += 1
    fix_ndx = np.array(fix_ndx)

    return ckpt_ndx, fix_ndx


def _our_method(explainer, noisy_ndx, y_train, n_check, interval):
    """
    Order instnces by largest absolute values of TREX weights.
    """
    train_weight = explainer.get_weight()[0]
    train_order = np.argsort(np.abs(train_weight))[::-1][:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, len(y_train), interval)

    return ckpt_ndx, fix_ndx, train_weight


def _random_method(noisy_ndx, y_train, interval, to_check=1, random_state=1):
    """
    Randomly picks train instances from the train data.
    """

    n_train = len(y_train)

    assert to_check <= n_train, 'to_check > n_train!'
    if isinstance(to_check, int):
        n_check = to_check
    else:
        exit('to_check not int')

    np.random.seed(random_state + 1)  # +1 to avoid choosing the same indices as the noisy labels
    train_order = np.random.choice(n_train, size=n_check, replace=False)
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx


def _loss_method(noisy_ndx, y_train_proba, y_train, interval, to_check=1, logloss=False):
    """
    Sorts train instances by largest train loss.
    """

    n_train = len(y_train)

    assert to_check <= n_train, 'to_check > n_train!'
    if isinstance(to_check, int):
        n_check = to_check
    else:
        exit('to_check not int')

    # extract 1d array of probabilities representing the probability of the target label
    if y_train_proba.ndim > 1:
        y_proba = model_util.positive_class_proba(y_train, y_train_proba)
    else:
        y_proba = y_train_proba

    # compute the loss for each instance
    y_loss = exp_util.instance_loss(y_proba, y_train, logloss=logloss)

    # put train instances in order based on decreasing absolute loss
    if logloss:
        train_order = np.argsort(y_loss)[:n_check]  # ascending order, most negative first
    else:
        train_order = np.argsort(y_loss)[::-1][:n_check]  # descending order, most positive first

    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, len(y_train), interval)
    return ckpt_ndx, fix_ndx, y_loss, train_order


def _influence_method(explainer, noisy_ndx, X_train, y_train, y_train_noisy, interval, to_check=1):
    """
    Computes the influence on train instance i if train instance i were upweighted/removed.
    Reference: https://github.com/kohpangwei/influence-release/blob/master/influence/experiments.py
    This uses the fastleafinfluence method by Sharchilev et al.
    """

    n_train = len(y_train)

    assert to_check <= n_train, 'to_check > n_train!'
    if isinstance(to_check, int):
        n_check = to_check
    else:
        exit('to_check not int')

    influence_scores = []
    buf = deepcopy(explainer)
    for i in tqdm.tqdm(range(len(X_train))):
        explainer.fit(removed_point_idx=i, destination_model=buf)
        influence_scores.append(buf.loss_derivative(X_train[[i]], y_train_noisy[[i]])[0])
    influence_scores = np.array(influence_scores)

    # sort by ascending order; the most negative train instances
    # are the ones that increase the log loss the most, and are the most harmful
    train_order = np.argsort(influence_scores)[:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx, influence_scores, train_order


def _maple_method(explainer, X_train, noisy_ndx, interval, to_check=1):
    """
    Orders instances by tree kernel similarity density.
    """

    n_train = X_train.shape[0]

    assert to_check <= n_train, 'to_check > n_train!'
    if isinstance(to_check, int):
        n_check = to_check
    else:
        exit('to_check not int')

    train_weight = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_train.shape[0])):
        weights = explainer.get_weights(X_train[i])
        train_weight[i] += np.sum(weights)

    train_order = np.argsort(train_weight)[::-1][:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx, train_weight, train_order


def _knn_method(knn_clf, X_train, noisy_ndx, interval, to_check=1):
    """
    Count impact by the number of times a training sample shows up in
    one another's neighborhoods, this can be weighted by 1 / distance.
    """

    n_train = X_train.shape[0]

    assert to_check <= n_train, 'to_check > n_train!'
    if isinstance(to_check, int):
        n_check = to_check
    else:
        exit('to_check not int')

    train_impact = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_train.shape[0])):
        distances, neighbor_ids = knn_clf.kneighbors([X_train[i]])
        train_impact[neighbor_ids[0]] += 1

    train_order = np.argsort(train_impact)[::-1][:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx, train_order


def _proto_method(model, X_train, y_train, noisy_ndx, interval, n_check):
    """
    Orders instances by using the GBT distance similarity formula in
    https://arxiv.org/pdf/1611.07115.pdf, then ranks training samples
    based on the proportion of labels from the k = 10, nearest neighbors.
    """
    extractor = trex.TreeExtractor(model, tree_kernel='leaf_path')
    X_train_alt = extractor.fit_transform(X_train)

    # obtain weight of each tree: note, this code is specific to CatBoost
    temp_fp = '.{}_cb.json'.format(str(uuid.uuid4()))
    model.save_model(temp_fp, format='json')
    cb_dump = json.load(open(temp_fp, 'r'))

    # obtain weight of each tree: learning_rate^2 * var(predictions)
    tree_weights = []
    for tree in cb_dump['oblivious_trees']:
        predictions = []

        for val, weight in zip(tree['leaf_values'], tree['leaf_weights']):
            predictions += [val] * weight

        tree_weights.append(np.var(predictions) * (model.learning_rate_ ** 2))

    # weight leaf path feature representation by the tree weights
    for i in range(X_train_alt.shape[0]):

        weight_cnt = 0
        for j in range(X_train_alt.shape[1]):

            if X_train_alt[i][j] == 1:
                X_train_alt[i][j] *= tree_weights[weight_cnt]
                weight_cnt += 1

        assert weight_cnt == len(tree_weights)

    # build a KNN using this proximity measure using k = 10
    knn = KNeighborsClassifier(n_neighbors=TREE_PROTO_K)
    knn = knn.fit(X_train_alt, y_train)

    # compute proportion of neighbors that share the same label
    train_impact = np.zeros(X_train_alt.shape[0])
    for i in tqdm.tqdm(range(X_train_alt.shape[0])):
        _, neighbor_ids = knn.kneighbors([X_train_alt[i]])
        train_impact[i] = len(np.where(y_train[i] == y_train[neighbor_ids[0]])[0]) / len(neighbor_ids[0])

    # rank training instances by low label agreement with its neighbors
    train_order = np.argsort(train_impact)[:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_check, interval)

    os.system('rm {}'.format(temp_fp))

    return ckpt_ndx, fix_ndx


def _mmd_method(model, X_train, y_train, noisy_ndx, interval, n_check):
    """
    Orders instances by prototypes and/or criticisms.
    """
    n_prototypes = N_PROTOTYPES
    n_criticisms = n_check

    X = np.hstack([X_train, y_train.reshape(-1, 1)])
    K = rbf_kernel(X)
    candidates = np.arange(len(K))

    prototypes = mmd.greedy_select_protos(K, candidates, n_prototypes)
    criticisms = mmd.select_criticism_regularized(K, prototypes, n_criticisms, is_K_sparse=False)

    train_order = criticisms[:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, len(y_train), interval)

    return ckpt_ndx, fix_ndx


def experiment(args, logger, out_dir, seed):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=seed)

    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset,
                                                                 random_state=seed,
                                                                 data_dir=args.data_dir)

    # reduce train size
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train = int(X_train.shape[0] * args.train_frac)
        X_train, y_train = X_train[:n_train], y_train[:n_train]
    data = X_train, y_train, X_test, y_test

    logger.info('no. train instances: {:,}'.format(len(X_train)))
    logger.info('no. test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # add noise
    y_train_noisy, noisy_ndx = data_util.flip_labels(y_train, k=args.flip_frac, random_state=seed)
    noisy_ndx = np.array(sorted(noisy_ndx))
    logger.info('no. noisy labels: {:,}'.format(len(noisy_ndx)))

    # train a tree ensemble on the clean and noisy labels
    model = clone(clf).fit(X_train, y_train)
    model_noisy = clone(clf).fit(X_train, y_train_noisy)

    # show model performance before and after noise
    logger.info('\nBefore noise:')
    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)
    logger.info('\nAfter noise:')
    model_util.performance(model_noisy, X_train, y_train_noisy, X_test=X_test, y_test=y_test, logger=logger)

    # check accuracy before and after noise
    acc_test_clean = accuracy_score(y_test, model.predict(X_test))
    acc_test_noisy = accuracy_score(y_test, model_noisy.predict(X_test))

    # find how many corrupted/non-corrupted labels were incorrectly predicted
    if not args.true_label:
        logger.info('\nUsing predicted labels:')
        predicted_labels = model_noisy.predict(X_train).flatten()
        incorrect_ndx = np.where(y_train_noisy != predicted_labels)[0]
        incorrect_corrupted_ndx = np.intersect1d(noisy_ndx, incorrect_ndx)
        logger.info('incorrectly predicted corrupted labels: {:,}'.format(incorrect_corrupted_ndx.shape[0]))
        logger.info('total number of incorrectly predicted labels: {:,}'.format(incorrect_ndx.shape[0]))

    # number of checkpoints to record
    n_check = int(len(y_train) * args.check_pct)
    interval = (n_check / len(y_train)) / args.n_plot_points

    # random method
    logger.info('\nordering by random...')
    start = time.time()
    ckpt_ndx, fix_ndx = _random_method(noisy_ndx, y_train, interval,
                                       to_check=n_check,
                                       random_state=seed)
    check_pct, random_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
    logger.info('time: {:3f}s'.format(time.time() - start))
    np.save(os.path.join(out_dir, 'random.npy'), random_res)

    # save global lines
    np.save(os.path.join(out_dir, 'test_clean.npy'), acc_test_clean)
    np.save(os.path.join(out_dir, 'check_pct.npy'), check_pct)

    # tree loss method
    logger.info('\nordering by tree loss...')
    start = time.time()

    y_train_proba = model_noisy.predict_proba(X_train)
    ckpt_ndx, fix_ndx, _, _ = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval, to_check=n_check)
    _, tree_loss_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    logger.info('time: {:3f}s'.format(time.time() - start))
    np.save(os.path.join(out_dir, 'tree.npy'), tree_loss_res)

    # trex method
    if args.trex:
        logger.info('\nordering by TREX...')
        start = time.time()
        explainer = trex.TreeExplainer(model_noisy, X_train, y_train_noisy,
                                       tree_kernel=args.tree_kernel,
                                       random_state=seed,
                                       true_label=args.true_label,
                                       kernel_model=args.kernel_model,
                                       verbose=args.verbose,
                                       val_frac=args.val_frac,
                                       logger=logger)

        ckpt_ndx, fix_ndx, _ = _our_method(explainer, noisy_ndx, y_train, n_check, interval)
        check_pct, trex_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), trex_res)

        # trex loss method
        logger.info('\nordering by TREX loss...')
        start = time.time()

        y_train_proba = explainer.predict_proba(X_train)
        ckpt_ndx, fix_ndx, _, _ = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval, to_check=n_check)
        _, trex_loss_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method_loss.npy'), trex_loss_res)

    # influence method
    if args.tree_type == 'cb' and args.inf_k is not None:
        logger.info('\nordering by leafinfluence...')
        start = time.time()

        model_path = '.model.json'
        model_noisy.save_model(model_path, format='json')

        if args.inf_k == -1:
            update_set = 'AllPoints'
        elif args.inf_k == 0:
            update_set = 'SinglePoint'
        else:
            update_set = 'TopKLeaves'

        leaf_influence = CBLeafInfluenceEnsemble(model_path, X_train, y_train_noisy, k=args.inf_k,
                                                 learning_rate=model.learning_rate_, update_set=update_set)
        ckpt_ndx, fix_ndx, _, _ = _influence_method(leaf_influence, noisy_ndx, X_train, y_train, y_train_noisy,
                                                    interval, to_check=n_check)
        _, leafinfluence_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), leafinfluence_res)

    # MAPLE method
    if args.maple:
        logger.info('\nordering by MAPLE...')
        start = time.time()

        train_label = y_train_noisy if args.true_label else model_noisy.predict(X_train)
        maple_exp = MAPLE(X_train, train_label, X_train, train_label, verbose=args.verbose, dstump=False)
        ckpt_ndx, fix_ndx, map_scores, map_order = _maple_method(maple_exp, X_train, noisy_ndx, interval,
                                                                 to_check=n_check)
        _, maple_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), maple_res)

    # TEKNN method
    if args.teknn:
        logger.info('\nordering by teknn...')
        start = time.time()

        # transform the data
        extractor = trex.TreeExtractor(model_noisy, tree_kernel=args.tree_kernel)
        X_train_alt = extractor.fit_transform(X_train)
        train_label = y_train if args.true_label else model_noisy.predict(X_train)

        # tune and train teknn
        knn_clf = exp_util.tune_knn(model_noisy, X_train, X_train_alt, train_label, args.val_frac,
                                    seed=seed, logger=logger)

        ckpt_ndx, fix_ndx, _ = _knn_method(knn_clf, X_train_alt, noisy_ndx, interval, to_check=n_check)
        _, teknn_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), teknn_res)

        # TEKNN loss method
        logger.info('\nordering by teknn loss...')
        start = time.time()
        y_train_proba = knn_clf.predict_proba(X_train_alt)

        ckpt_ndx, fix_ndx, _, _ = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval, to_check=n_check)
        _, teknn_loss_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method_loss.npy'), teknn_loss_res)

    # MMD-Critic method
    if args.mmd:
        logger.info('\nordering by mmd-critic...')
        start = time.time()
        ckpt_ndx, fix_ndx = _mmd_method(model_noisy, X_train, y_train_noisy, noisy_ndx, interval, n_check)
        _, mmd_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), mmd_res)

    # Prototype method
    if args.proto:
        logger.info('\nordering by proto...')
        start = time.time()
        ckpt_ndx, fix_ndx = _proto_method(model_noisy, X_train, y_train_noisy, noisy_ndx, interval, n_check)
        _, proto_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), proto_res)


def main(args):

    # make logger
    dataset = args.dataset

    if args.train_frac < 1.0 and args.train_frac > 0.0:
        dataset += '_{}'.format(str(args.train_frac).replace('.', 'p'))

    out_dir = os.path.join(args.out_dir, dataset, args.tree_type,
                           'rs{}'.format(args.rs))

    if args.trex:
        out_dir = os.path.join(out_dir, args.kernel_model, args.tree_kernel)
    elif args.teknn:
        out_dir = os.path.join(out_dir, 'teknn', args.tree_kernel)
    elif args.maple:
        out_dir = os.path.join(out_dir, 'maple')
    elif args.inf_k is not None:
        out_dir = os.path.join(out_dir, 'leaf_influence')
    elif args.mmd:
        out_dir = os.path.join(out_dir, 'mmd')
    elif args.proto:
        out_dir = os.path.join(out_dir, 'proto')

    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    seed = args.rs
    logger.info('\nSeed: {}'.format(seed))
    experiment(args, logger, out_dir, seed=seed)
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O settings
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/cleaning/', help='output directory.')

    # data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='amount of training data to use.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')
    parser.add_argument('--flip_frac', type=float, default=0.4, help='fraction of train labels to flip.')

    # tree settings
    parser.add_argument('--tree_type', type=str, default='cb', help='tree model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')

    # TREX settings
    parser.add_argument('--trex', action='store_true', default=False, help='use TREX.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--true_label', action='store_true', default=False, help='train model on true labels.')
    parser.add_argument('--kernel_model', type=str, default='klr', help='kernel model to use.')

    # method settings
    parser.add_argument('--inf_k', type=int, default=None, help='number of leaves to use for leafinfluence.')
    parser.add_argument('--maple', action='store_true', default=False, help='whether to use MAPLE as a baseline.')
    parser.add_argument('--teknn', action='store_true', default=False, help='use KNN on top of TREX features.')
    parser.add_argument('--mmd', action='store_true', default=False, help='MMD-Critic prototypes.')
    parser.add_argument('--proto', action='store_true', default=False, help='Tree prototypes.')

    # plot settings
    parser.add_argument('--check_pct', type=float, default=0.3, help='max percentage of train instances to check.')
    parser.add_argument('--n_plot_points', type=int, default=10, help='number of points to plot.')

    # experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
