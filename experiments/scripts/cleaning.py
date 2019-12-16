"""
Experiment: dataset cleaning from noisy labels, specifically a percentage of flipped labels
in the train data. Only for binary classification datasets.
"""
import time
import argparse
from copy import deepcopy
import os
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale

import trex
from utility import model_util, data_util, exp_util, print_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from maple.MAPLE import MAPLE


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

    train_weight = explainer.get_weight()[0]
    train_order = np.argsort(np.abs(train_weight))[::-1][:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, len(y_train), interval)

    return ckpt_ndx, fix_ndx, train_weight


def _random_method(noisy_ndx, y_train, interval, to_check=1, random_state=1):
    """Randomly picks train instances from the train data."""

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
    """Sorts train instances by largest train loss."""

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

    # sort by descending order; the most negative train instances
    # are the ones that increase the log loss the most, and are the most harmful
    train_order = np.argsort(influence_scores)[:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx, influence_scores, train_order


def _maple_method(explainer, X_train, noisy_ndx, interval, to_check=1):

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


def _knn_method(knn_clf, weights, X_train, noisy_ndx, interval, to_check=1):
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
        if weights == 'uniform':
            train_impact[neighbor_ids[0]] += 1
        else:
            train_impact[neighbor_ids[0]] += 1 / distances[0]

    train_order = np.argsort(train_impact)[::-1][:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx, train_order


def noise_detection(args, logger, out_dir, seed=1):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # get model and data
    clf = model_util.get_classifier(args.model_type, n_estimators=args.n_estimators, max_depth=args.max_depth,
                                    random_state=seed)
    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=seed,
                                                                 data_dir=args.data_dir)

    # reduce train size
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train = int(X_train.shape[0] * args.train_frac)
        X_train, y_train = X_train[:n_train], y_train[:n_train]
    data = X_train, y_train, X_test, y_test

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('test instances: {}'.format(len(X_test)))

    # add noise
    y_train_noisy, noisy_ndx = data_util.flip_labels(y_train, k=args.flip_frac, random_state=seed)
    noisy_ndx = np.array(sorted(noisy_ndx))
    logger.info('num noisy labels: {}'.format(len(noisy_ndx)))

    # use part of the test data as validation data
    X_val = X_test.copy()
    if args.val_frac < 1.0 and args.val_frac > 0.0:
        X_val = X_val[int(X_val.shape[0] * args.val_frac):]

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
        logger.info('incorrectly predicted corrupted labels: {}'.format(incorrect_corrupted_ndx.shape[0]))
        logger.info('total number of incorrectly predicted labels: {}'.format(incorrect_ndx.shape[0]))

    # number of checkpoints to record
    n_check = int(len(y_train) * args.check_pct)
    interval = (n_check / len(y_train)) / args.n_plot_points

    # random method
    logger.info('ordering by random...')
    start = time.time()
    ckpt_ndx, fix_ndx = _random_method(noisy_ndx, y_train, interval, to_check=n_check, random_state=seed)
    check_pct, random_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
    logger.info('time: {:3f}s'.format(time.time() - start))

    # tree loss method
    logger.info('ordering by tree loss...')
    start = time.time()
    y_train_proba = model_noisy.predict_proba(X_train)
    ckpt_ndx, fix_ndx, _, _ = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval, to_check=n_check)
    _, tree_loss_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
    logger.info('time: {:3f}s'.format(time.time() - start))

    # our method
    if args.trex:
        logger.info('ordering by our method...')
        start = time.time()
        explainer = trex.TreeExplainer(model_noisy, X_train, y_train_noisy, encoding=args.encoding, dense_output=True,
                                       random_state=seed, use_predicted_labels=not args.true_label,
                                       kernel=args.kernel, linear_model=args.linear_model, C=args.C,
                                       verbose=args.verbose, X_val=X_val)
        logger.info('C: {}'.format(explainer.C_))
        ckpt_ndx, fix_ndx, _ = _our_method(explainer, noisy_ndx, y_train, n_check, interval)
        check_pct, our_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
        settings = '{}_{}'.format(args.linear_model, args.kernel)
        settings += '_true_label' if args.true_label else ''
        logger.info('time: {:3f}s'.format(time.time() - start))

    # linear loss method - if svm, squish decision values to between 0 and 1
    if args.trex and args.linear_model_loss:
        logger.info('ordering by linear loss...')
        start = time.time()
        if args.linear_model == 'svm':
            y_train_proba = explainer.decision_function(X_train)
            if y_train_proba.ndim == 1:
                y_train_proba = exp_util.make_multiclass(y_train_proba)
            y_train_proba = minmax_scale(y_train_proba)
        else:
            y_train_proba = explainer.predict_proba(X_train)
        ckpt_ndx, fix_ndx, _, _ = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval, to_check=n_check)
        _, linear_loss_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # influence method
    if args.model_type == 'cb' and args.inf_k is not None:
        logger.info('ordering by leafinfluence...')
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

    # MAPLE method
    if args.maple:
        logger.info('ordering by MAPLE...')
        start = time.time()
        train_label = y_train if args.true_label else model_noisy.predict(X_train)
        maple_exp = MAPLE(X_train, train_label, X_train, y_train_noisy, verbose=args.verbose, dstump=False)
        ckpt_ndx, fix_ndx, map_scores, map_order = _maple_method(maple_exp, X_train, noisy_ndx, interval,
                                                                 to_check=n_check)
        _, maple_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # KNN method
    if args.knn:
        logger.info('ordering by knn...')
        start = time.time()

        # transform the data
        extractor = trex.TreeExtractor(model_noisy, encoding=args.encoding)
        X_train_alt = extractor.fit_transform(X_train)
        X_val_alt = extractor.transform(X_val)
        train_label = y_train if args.true_label else model_noisy.predict(X_train)

        # tune and train teknn
        knn_clf, params = exp_util.tune_knn(X_train_alt, train_label, model_noisy, X_val, X_val_alt)
        logger.info('n_neighbors: {}, weights: {}'.format(params['n_neighbors'], params['weights']))
        weights = params['weights']

        ckpt_ndx, fix_ndx, _ = _knn_method(knn_clf, weights, X_train_alt, noisy_ndx, interval, to_check=n_check)
        _, knn_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # knn loss method
    if args.knn and args.knn_loss:
        logger.info('ordering by knn loss...')
        start = time.time()
        y_train_proba = knn_clf.predict_proba(X_train_alt)
        ckpt_ndx, fix_ndx, _, _ = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval, to_check=n_check)
        _, knn_loss_res = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # plot results
    logger.info('plotting...')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(check_pct, random_res, marker='^', color='r', label='random')
    ax.plot(check_pct, tree_loss_res, marker='p', color='c', label='tree_loss')
    ax.axhline(acc_test_clean, color='k', linestyle='--')
    ax.set_xlabel('fraction of train data checked')
    ax.set_ylabel('test accuracy')
    if args.trex:
        ax.plot(check_pct, our_res, marker='.', color='g', label='ours')
    if args.trex and args.linear_model_loss:
        ax.plot(check_pct, linear_loss_res, marker='*', color='y', label='linear_loss')
    if args.model_type == 'cb' and args.inf_k is not None:
        ax.plot(check_pct, leafinfluence_res, marker='+', color='m', label='leafinfluence')
    if args.maple:
        ax.plot(check_pct, maple_res, marker='o', color='orange', label='maple')
    if args.knn:
        ax.plot(check_pct, knn_res, marker='D', color='magenta', label='knn')
    if args.knn and args.knn_loss:
        ax.plot(check_pct, knn_loss_res, marker='h', color='#EEC64F', label='knn_loss')
    ax.legend()

    if args.save_results:

        # save plot
        rs_dir = os.path.join(out_dir, 'rs{}'.format(seed))
        os.makedirs(rs_dir, exist_ok=True)

        # save plot
        logger.info('saving plot...')
        plt.savefig(os.path.join(rs_dir, 'cleaning.pdf'), format='pdf', bbox_inches='tight')

        # save global lines
        np.save(os.path.join(rs_dir, 'test_clean.npy'), acc_test_clean)
        np.save(os.path.join(rs_dir, 'check_pct.npy'), check_pct)

        # ours, random, and tree loss
        np.save(os.path.join(rs_dir, 'random.npy'), random_res)
        np.save(os.path.join(rs_dir, 'tree_loss.npy'), tree_loss_res)

        # trex
        if args.trex:
            np.save(os.path.join(rs_dir, 'our_{}.npy'.format(settings)), our_res)

        # linear model loss
        if args.trex and args.linear_model_loss:
            np.save(os.path.join(rs_dir, '{}_loss.npy'.format(settings)), linear_loss_res)

        # leaf influence
        if args.model_type == 'cb' and args.inf_k is not None:
            np.save(os.path.join(rs_dir, 'leafinfluence.npy'), leafinfluence_res)

        # maple
        if args.maple:
            np.save(os.path.join(rs_dir, 'maple.npy'), maple_res)

        # knn
        if args.knn:
            np.save(os.path.join(rs_dir, 'knn.npy'), knn_res)

        # knn loss
        if args.knn and args.knn_loss:
            np.save(os.path.join(rs_dir, 'knn_loss.npy'), knn_loss_res)

    if args.show_plot:
        plt.show()


def main(args):

    # make logger
    dataset = args.dataset
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        dataset += '_{}'.format(args.dataset, str(args.train_frac).replace('.', 'p'))
    out_dir = os.path.join(args.out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    seed = args.rs
    for i in range(args.repeats):
        logger.info('\nRun {}, seed: {}'.format(i + 1, seed))
        noise_detection(args, logger, out_dir, seed=seed)
        seed += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/cleaning', help='output directory.')
    parser.add_argument('--train_frac', type=float, default=1.0, help='amount of training data to use.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')
    parser.add_argument('--repeats', type=int, default=1, help='repeats of the experiment.')
    parser.add_argument('--model_type', type=str, default='cb', help='tree model to use.')
    parser.add_argument('--trex', action='store_true', help='Use TREX.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')
    parser.add_argument('--kernel', default='linear', help='Similarity kernel for the linear model.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=1, help='for reproducibility.')
    parser.add_argument('--linear_model_loss', action='store_true', default=False, help='Include linear loss.')
    parser.add_argument('--show_plot', action='store_true', default=False, help='Save plot results.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--flip_frac', type=float, default=0.4, help='Fraction of train labels to flip.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--maple', action='store_true', help='Whether to use MAPLE as a baseline.')
    parser.add_argument('--check_pct', type=float, default=0.3, help='Max percentage of train instances to check.')
    parser.add_argument('--n_plot_points', type=int, default=10, help='Number of points to plot.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--knn_loss', action='store_true', default=False, help='Use KNN loss method.')
    args = parser.parse_args()
    main(args)
