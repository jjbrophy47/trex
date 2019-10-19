"""
Experiment: Cleaning experiment with medifor datasets, this one:
    1. generates an ordering.
    2. checks the first w%, fixing labels if they were flipped.
    3. retrains the model on the semi noisy data.
    4. repeat 1-3 x times.
"""
import argparse
import os
import sys
from copy import deepcopy
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for trex

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

import trex
from utility import model_util, data_util, exp_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble


def _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy, scoring='accuracy'):
    """
    Retrains the tree ensemble for each ckeckpoint, where a checkpoint represents
    which flipped labels have been fixed.
    """

    X_train, y_train, X_test, y_test = data

    accs = []
    checked_pct = []
    fix_pct = []

    for n_checked, n_fix in tqdm.tqdm(ckpt_ndx):
        fix_indices = fix_ndx[:n_fix]

        semi_noisy_ndx = np.setdiff1d(noisy_ndx, fix_indices)
        y_train_semi_noisy = data_util.flip_labels_with_indices(y_train, semi_noisy_ndx)

        model_semi_noisy = clone(clf).fit(X_train[semi_noisy_ndx], y_train_semi_noisy[semi_noisy_ndx])

        if scoring == 'accuracy':
            acc_test = accuracy_score(y_test, model_semi_noisy.predict(X_test))
        else:
            acc_test = roc_auc_score(y_test, model_semi_noisy.predict_proba(X_test)[:, 1])

        accs.append(acc_test)
        checked_pct.append(float(n_checked / len(y_train)))
        fix_pct.append(float(n_fix / len(noisy_ndx)))

    return checked_pct, accs, fix_pct


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


def _our_method(explainer):
    """Sorts train instances by largest weight."""

    train_weight = explainer.get_weight()[0]
    train_order = np.argsort(np.abs(train_weight))[::-1]
    return train_order


def _random_method(n_train, iteration, random_state=69):
    """Randomly picks train instances from the train data."""

    np.random.seed(random_state + iteration)  # +iteration to avoid choosing the same ordering
    train_order = np.random.choice(n_train, size=n_train, replace=False)
    return train_order


def _loss_method(y_train_proba, y_train, logloss=False):
    """Sorts train instances by largest train loss."""

    # extract 1d array of probabilities representing the probability of the target label
    if y_train_proba.ndim > 1:
        y_proba = model_util.positive_class_proba(y_train, y_train_proba)
    else:
        y_proba = y_train_proba

    # compute the loss for each instance
    y_loss = exp_util.instance_loss(y_proba, y_train, logloss=logloss)

    # put train instances in order based on decreasing absolute loss
    if logloss:
        train_order = np.argsort(y_loss)  # ascending order, most negative first
    else:
        train_order = np.argsort(y_loss)[::-1]  # descending order, most positive first
    return train_order


def _linear_model_loss_method(explainer, linear_model, X_train, y_train_noisy):
    """linear loss method - if smv, squish decision values to between 0 and 1"""

    if linear_model == 'svm':
        y_train_proba = explainer.decision_function(X_train)
        if y_train_proba.ndim == 1:
            y_train_proba = exp_util.make_multiclass(y_train_proba)
        y_train_proba = minmax_scale(y_train_proba)
    else:
        y_train_proba = explainer.predict_proba(X_train)

    train_order = _loss_method(y_train_proba, y_train_noisy)

    return train_order


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


def noise_detection(model_type='lgb', encoding='leaf_output', dataset='nc17_mfc18', linear_model='svm',
                    n_estimators=100, random_state=69, inf_k=None, data_dir='data', scoring='accuracy',
                    true_label=False, kernel='linear', out_dir='output/cleaning', linear_model_loss=False,
                    save_plot=False, save_results=False, alpha=0.69, check_pct=0.3, values_pct=0.1,
                    num_iterations=5):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # get model and data
    clf = model_util.get_classifier(model_type, n_estimators=n_estimators, random_state=random_state)
    X_pre_train, X_test, y_pre_train, y_test, label = data_util.get_data(dataset, random_state=random_state,
                                                                         data_dir=data_dir)

    # split the test set into two pieces: m1 and m2
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.5,
                                                        random_state=random_state, stratify=y_test)

    print('\npre-train instances: {}'.format(len(X_pre_train)))
    print('train instances: {}'.format(len(X_train)))
    print('test instances: {}\n'.format(len(X_test)))

    # train a tree ensemble on the training set
    tree = clone(clf).fit(X_pre_train, y_pre_train)
    model_util.performance(tree, X_pre_train, y_pre_train, X_train, y_train)

    # train a clean model on m1
    model_clean = clone(clf).fit(X_train, y_train)
    print('\nGround-Truth Performance:')
    model_util.performance(model_clean, X_train, y_train, X_test, y_test)

    # generate predictions p1 for m1
    y_train_noisy = tree.predict(X_train).astype(int)

    # get indices of noisy labels (incorrectly predicted labels)
    noisy_ndx = np.where(y_train_noisy != y_train)[0]
    noisy_ndx = np.array(sorted(noisy_ndx))
    print('\nnum noisy labels: {}'.format(len(noisy_ndx)))

    n_check = int(check_pct * len(y_train))
    y_train_semi_noisy = y_train_noisy.copy()
    fixed_indices = []

    for i in range(num_iterations):

        # train a model on m1 using p1
        model_noisy = clone(clf).fit(X_train, y_train_semi_noisy)
        print('\nNoisy Performance:')
        model_util.performance(model_noisy, X_train, y_train_semi_noisy, X_test, y_test)

        # our method
        print('ordering...')
        explainer = trex.TreeExplainer(model_noisy, X_train, y_train_semi_noisy, encoding=encoding, dense_output=True,
                                       random_state=random_state, use_predicted_labels=not true_label,
                                       kernel=kernel, linear_model=linear_model)
        train_order = _our_method(explainer)
        # train_order = _random_method(len(y_train), iteration=i, random_state=random_state)
        # train_order = _loss_method(model_noisy.predict_proba(X_train), y_train_semi_noisy)
        # train_order = _linear_model_loss_method(explainer, linear_model, X_train, y_train_semi_noisy)
        fix_indices = [ndx for ndx in train_order[:n_check] if ndx in noisy_ndx and ndx not in fixed_indices]
        fixed_indices += fix_indices
        semi_noisy_ndx = np.setdiff1d(noisy_ndx, fixed_indices)
        y_train_semi_noisy = data_util.flip_labels_with_indices(y_train, semi_noisy_ndx)

    # train a model on m1 using p1
    model_noisy = clone(clf).fit(X_train, y_train_semi_noisy)
    print('\nNoisy Performance:')
    model_util.performance(model_noisy, X_train, y_train_semi_noisy, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='tree model to use.')
    parser.add_argument('--linear_model', type=str, default='svm', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--scoring', type=str, default='accuracy', help='metric to use for scoring.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--kernel', default='linear', help='Similarity kernel for the linear model.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--linear_model_loss', action='store_true', help='compare with the linear model loss.')
    parser.add_argument('--inf_k', type=int, default=None, help='compare with leafinfluence.')
    parser.add_argument('--save_plot', action='store_true', default=False, help='Save plot results.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--check_pct', type=float, default=0.3, help='Max percentage of train instances to check.')
    parser.add_argument('--values_pct', type=float, default=0.1, help='Percentage of weights/loss values to compare.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    args = parser.parse_args()
    print(args)
    noise_detection(model_type=args.model, encoding=args.encoding, dataset=args.dataset, values_pct=args.values_pct,
                    linear_model=args.linear_model, kernel=args.kernel, check_pct=args.check_pct, inf_k=args.inf_k,
                    n_estimators=args.n_estimators, random_state=args.rs,
                    save_plot=args.save_plot, scoring=args.scoring, linear_model_loss=args.linear_model_loss,
                    save_results=args.save_results, true_label=args.true_label)
