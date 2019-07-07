"""
Experiment: dataset cleaning from noisy labels, specifically a percentage of flipped labels
in the train data. Only for binary classification datasets.
"""
import argparse
from copy import deepcopy
import os
import sys
sys.path.insert(0, os.getcwd())  # for influence_boosting

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale

import sexee
from util import model_util, data_util, exp_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble


def interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy):
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

    return checked_pct, accs, fix_pct


def record_fixes(train_order, noisy_ndx, train_len, interval):
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


def sexee_method(explainer, noisy_ndx, y_train, points=10):
    """Sorts train instances by largest weight support vectors."""

    train_weight = explainer.get_train_weight()
    n_check = len(train_weight)
    train_order = [train_ndx for train_ndx, weight in train_weight]
    interval = (n_check / len(y_train)) / points
    ckpt_ndx, fix_ndx = record_fixes(train_order, noisy_ndx, len(y_train), interval)

    return ckpt_ndx, fix_ndx, interval, n_check


def sexee_method2(explainer, noisy_ndx, X_train, y_train, points=10):
    """Sorts train instances by largest total absolute impact on the train set."""

    train_ndx, impact = explainer.train_impact(X_train)
    impact = np.sum(impact, axis=1)
    assert len(train_ndx) == len(impact)

    impact_list = zip(train_ndx, impact)
    impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)
    train_order = [train_impact[0] for train_impact in impact_list]

    n_check = len(train_order)
    interval = (n_check / len(y_train)) / points
    ckpt_ndx, fix_ndx = record_fixes(train_order, noisy_ndx, len(y_train), interval)

    return ckpt_ndx, fix_ndx, interval, n_check


def random_method(noisy_ndx, y_train, interval, to_check=1, random_state=69):
    """Randomly picks train instances from the train data."""

    n_train = len(y_train)

    assert to_check <= n_train, 'to_check > n_train!'
    if isinstance(to_check, int):
        n_check = to_check
    else:
        exit('to_check not int')

    np.random.seed(random_state + 1)  # +1 to avoid choosing the same indices as the noisy labels
    train_order = np.random.choice(n_train, size=n_check, replace=False)
    ckpt_ndx, fix_ndx = record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx


def loss_method(noisy_ndx, y_train_proba, y_train, interval, to_check=1, logloss=False):
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

    ckpt_ndx, fix_ndx = record_fixes(train_order, noisy_ndx, len(y_train), interval)
    return ckpt_ndx, fix_ndx


def influence_method(explainer, noisy_ndx, X_train, y_train, y_train_noisy, interval, to_check=1):
    """
    Computes the influence on train instance i if train instance i were upweighted/removed.
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

    # sort by absolute value
    train_order = np.argsort(np.abs(influence_scores))[::-1][:n_check]
    ckpt_ndx, fix_ndx = record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx


def noise_detection(model_type='lgb', encoding='tree_path', dataset='iris', n_estimators=100, random_state=69,
                    timeit=False, inf_k=None, svm_loss=False, data_dir='data', flip_frac=0.4, true_label=False,
                    sexee2=False, out_dir='output/cleaning', save_plot=False):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # get model and data
    clf = model_util.get_classifier(model_type, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)
    data = X_train, y_train, X_test, y_test

    print('train instances: {}'.format(len(X_train)))
    print('test instances: {}'.format(len(X_test)))

    # add noise
    y_train_noisy, noisy_ndx = data_util.flip_labels(y_train, k=flip_frac, random_state=random_state)
    noisy_ndx = np.array(sorted(noisy_ndx))
    print('num noisy labels: {}'.format(len(noisy_ndx)))

    # train a tree ensemble on the clean and noisy labels
    model = clone(clf).fit(X_train, y_train)
    model_noisy = clone(clf).fit(X_train, y_train_noisy)

    # show model performance before and after noise
    print('\nBefore noise:')
    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test)
    print('\nAfter noise:')
    model_util.performance(model_noisy, X_train, y_train_noisy, X_test=X_test, y_test=y_test)

    # check accuracy before and after noise
    acc_test_clean = accuracy_score(y_test, model.predict(X_test))
    acc_test_noisy = accuracy_score(y_test, model_noisy.predict(X_test))

    # sexee method
    explainer = sexee.TreeExplainer(model_noisy, X_train, y_train_noisy, encoding=encoding,
                                    random_state=random_state, timeit=timeit, use_predicted_labels=not true_label)
    ckpt_ndx, fix_ndx, interval, n_check = sexee_method(explainer, noisy_ndx, y_train)
    sexee_results = interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # sexee method 2
    explainer = sexee.TreeExplainer(model_noisy, X_train, y_train_noisy, encoding=encoding,
                                    random_state=random_state, timeit=timeit, use_predicted_labels=not true_label)
    ckpt_ndx, fix_ndx, interval, n_check = sexee_method2(explainer, noisy_ndx, X_train, y_train)
    sexee2_results = interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # random method
    ckpt_ndx, fix_ndx = random_method(noisy_ndx, y_train, interval, to_check=n_check, random_state=random_state)
    random_results = interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # tree loss method
    y_train_proba = model_noisy.predict_proba(X_train)
    ckpt_ndx, fix_ndx = loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval, to_check=n_check)
    tree_loss_results = interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # svm loss method - squish svm decision values to between 0 and 1
    if svm_loss:
        y_train_proba = explainer.decision_function(X_train)
        if y_train_proba.ndim == 1:
            y_train_proba = exp_util.make_multiclass(y_train_proba)
        y_train_proba = minmax_scale(y_train_proba)
        ckpt_ndx, fix_ndx = loss_method(noisy_ndx, y_train_proba, y_train, interval, to_check=n_check)
        svm_loss_results = interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # influence method
    if model_type == 'cb' and inf_k is not None:
        model_path = '.model.json'
        model_noisy.save_model(model_path, format='json')

        if inf_k == -1:
            update_set = 'AllPoints'
        elif inf_k == 0:
            update_set = 'SinglePoint'
        else:
            update_set = 'TopKLeaves'

        leaf_influence = CBLeafInfluenceEnsemble(model_path, X_train, y_train_noisy,
                                                 learning_rate=model.learning_rate_, update_set=update_set, k=inf_k)
        ckpt_ndx, fix_ndx = influence_method(leaf_influence, noisy_ndx, X_train, y_train, y_train_noisy, interval,
                                             to_check=n_check)
        influence_results = interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # plot results
    sexee_check_pct, sexee_acc, sexee_fix_pct = sexee_results
    rand_check_pct, rand_acc, rand_fix_pct = random_results
    tree_loss_check_pct, tree_loss_acc, tree_loss_fix_pct = tree_loss_results

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(sexee_check_pct, sexee_acc, marker='.', color='g', label='ours')
    axs[0].plot(rand_check_pct, rand_acc, marker='^', color='r', label='random')
    axs[0].plot(tree_loss_check_pct, tree_loss_acc, marker='p', color='c', label='tree_loss')
    axs[0].axhline(acc_test_clean, color='k', linestyle='--')
    axs[0].set_xlabel('fraction of train data checked')
    axs[0].set_ylabel('test accuracy')
    axs[1].plot(sexee_check_pct, sexee_fix_pct, marker='.', color='g', label='ours')
    axs[1].plot(rand_check_pct, rand_fix_pct, marker='^', color='r', label='random')
    axs[1].plot(tree_loss_check_pct, tree_loss_fix_pct, marker='p', color='c', label='tree_loss')
    axs[1].set_xlabel('fraction of train data checked')
    axs[1].set_ylabel('fraction of flips fixed')
    if sexee2:
        sexee2_check_pct, sexee2_acc, sexee2_fix_pct = sexee2_results
        axs[0].plot(sexee2_check_pct, sexee2_acc, marker='<', color='orange', label='sexee2')
        axs[1].plot(sexee2_check_pct, sexee2_fix_pct, marker='<', color='orange', label='sexee2')
    if svm_loss:
        svm_loss_check_pct, svm_loss_acc, svm_loss_fix_pct = svm_loss_results
        axs[0].plot(svm_loss_check_pct, svm_loss_acc, marker='*', color='y', label='svm_loss')
        axs[1].plot(svm_loss_check_pct, svm_loss_fix_pct, marker='*', color='y', label='svm_loss')
    if model_type == 'cb' and inf_k is not None:
        influence_check_pct, influence_acc, influence_fix_pct = influence_results
        axs[0].plot(influence_check_pct, influence_acc, marker='+', color='m', label='leafinfluence')
        axs[1].plot(influence_check_pct, influence_fix_pct, marker='+', color='m', label='leafinfluence')
    axs[0].legend()
    axs[1].legend()

    if save_plot:
        plot_name = os.path.join(out_dir, dataset + '.pdf')
        print('saving to {}...'.format(plot_name))
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(plot_name, format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    parser.add_argument('--svm_loss', action='store_true', default=False, help='Include svm loss in results.')
    args = parser.parse_args()
    print(args)
    noise_detection(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.timeit, args.svm_loss)
