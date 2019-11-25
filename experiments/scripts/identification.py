"""
Experiment: Idenitfy mismatched instances by training on NC17, generating predictions
    for MFC18, retraining on the predicted labels of MFC18, then use TREX
    to order the training instances, and see how efficient we are at finding
    the misclassified instances.
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

    accs = [acc_test_noisy]
    checked_pct = [0]
    fix_pct = [0]

    for n_checked, n_fix in tqdm.tqdm(ckpt_ndx):
        fix_indices = fix_ndx[:n_fix]

        semi_noisy_ndx = np.setdiff1d(noisy_ndx, fix_indices)
        y_train_semi_noisy = data_util.flip_labels_with_indices(y_train, semi_noisy_ndx)

        model_semi_noisy = clone(clf).fit(X_train, y_train_semi_noisy)

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


def _our_method(explainer, noisy_ndx, y_train, points=10, cutoff_pct=0.3):
    """Sorts train instances by largest weight."""

    train_weight = explainer.get_weight()[0]
    n_check = int(len(y_train) * cutoff_pct)

    train_order = np.argsort(np.abs(train_weight))[::-1][:n_check]
    interval = (n_check / len(y_train)) / points
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, len(y_train), interval)

    return ckpt_ndx, fix_ndx, interval, n_check, train_weight


def _random_method(noisy_ndx, y_train, interval, to_check=1, random_state=69):
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


def noise_detection(model_type='lgb', encoding='leaf_output', dataset='nc17_mfc18', linear_model='svm',
                    n_estimators=100, random_state=69, inf_k=None, data_dir='data', scoring='accuracy',
                    true_label=False, kernel='linear', out_dir='output/cleaning', linear_model_loss=False,
                    save_plot=False, save_results=False, alpha=0.69, check_pct=0.3, values_pct=0.1):
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

    # train a model on m1 using p1
    model_noisy = clone(clf).fit(X_train, y_train_noisy)
    print('\nNoisy Performance:')
    model_util.performance(model_noisy, X_train, y_train_noisy, X_test, y_test)

    # get indices of noisy labels (incorrectly predicted labels)
    noisy_ndx = np.where(y_train_noisy != y_train)[0]
    noisy_ndx = np.array(sorted(noisy_ndx))
    print('\nnum noisy labels: {}'.format(len(noisy_ndx)))

    # train TREX on m1
    explainer = trex.TreeExplainer(model_noisy, X_train, y_train_noisy, encoding=encoding, dense_output=True,
                                   random_state=random_state, use_predicted_labels=not true_label,
                                   kernel=kernel, linear_model=linear_model)

    # check accuracy before and after noise
    if scoring == 'accuracy':
        acc_test_clean = accuracy_score(y_test, model_clean.predict(X_test))
        acc_test_noisy = accuracy_score(y_test, model_noisy.predict(X_test))
    else:
        acc_test_clean = roc_auc_score(y_test, model_clean.predict_proba(X_test)[:, 1])
        acc_test_noisy = roc_auc_score(y_test, model_noisy.predict_proba(X_test)[:, 1])

    # ordering training instances
    print('ordering by TREX...')
    data = X_train, y_train, X_test, y_test
    ckpt_ndx, fix_ndx, interval, n_check, our_train_weight = _our_method(explainer, noisy_ndx, y_train,
                                                                         cutoff_pct=check_pct)
    our_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy, scoring=scoring)
    settings = '{}_{}'.format(linear_model, kernel)
    settings += '_true_label' if true_label else ''

    # random method
    print('ordering by random...')
    ckpt_ndx, fix_ndx = _random_method(noisy_ndx, y_train, interval, to_check=n_check, random_state=random_state)
    random_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy, scoring=scoring)

    # tree loss method
    print('ordering by tree loss...')
    y_train_proba = model_noisy.predict_proba(X_train)
    ckpt_ndx, fix_ndx, tree_loss, tree_ndx = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval,
                                                          to_check=n_check)
    tree_loss_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy, scoring=scoring)

    # linear loss method - if smv, squish decision values to between 0 and 1
    if linear_model_loss:
        print('ordering by linear loss...')
        if linear_model == 'svm':
            y_train_proba = explainer.decision_function(X_train)
            if y_train_proba.ndim == 1:
                y_train_proba = exp_util.make_multiclass(y_train_proba)
            y_train_proba = minmax_scale(y_train_proba)
        else:
            y_train_proba = explainer.predict_proba(X_train)
        ckpt_ndx, fix_ndx, linear_loss, linear_ndx = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval,
                                                                  to_check=n_check)
        linear_loss_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy,
                                                    scoring=scoring)

    # influence method
    if model_type == 'cb' and inf_k is not None:
        print('ordering by leafinfluence...')

        model_path = '.model.json'
        model_noisy.save_model(model_path, format='json')

        if inf_k == -1:
            update_set = 'AllPoints'
        elif inf_k == 0:
            update_set = 'SinglePoint'
        else:
            update_set = 'TopKLeaves'

        leaf_influence = CBLeafInfluenceEnsemble(model_path, X_train, y_train_noisy,
                                                 learning_rate=model_noisy.learning_rate_, update_set=update_set,
                                                 k=inf_k)
        ckpt_ndx, fix_ndx, inf_scores, inf_order = _influence_method(leaf_influence, noisy_ndx, X_train, y_train,
                                                                     y_train_noisy, interval, to_check=n_check)
        influence_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy,
                                                  scoring=scoring)

    # plot results
    print('plotting...')
    our_check_pct, our_acc, our_fix_pct = our_results
    rand_check_pct, rand_acc, rand_fix_pct = random_results
    tree_loss_check_pct, tree_loss_acc, tree_loss_fix_pct = tree_loss_results

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(our_check_pct, our_acc, marker='.', color='g', label='ours')
    axs[0].plot(rand_check_pct, rand_acc, marker='^', color='r', label='random')
    axs[0].plot(tree_loss_check_pct, tree_loss_acc, marker='p', color='c', label='tree_loss')
    axs[0].axhline(acc_test_clean, color='k', linestyle='--')
    axs[0].set_xlabel('fraction of train data checked')
    axs[0].set_ylabel('test {}'.format(scoring))
    axs[1].plot(our_check_pct, our_fix_pct, marker='.', color='g', label='ours')
    axs[1].plot(rand_check_pct, rand_fix_pct, marker='^', color='r', label='random')
    axs[1].plot(tree_loss_check_pct, tree_loss_fix_pct, marker='p', color='c', label='tree_loss')
    axs[1].set_xlabel('fraction of train data checked')
    axs[1].set_ylabel('fraction of flips fixed')
    if linear_model_loss:
        linear_loss_check_pct, linear_loss_acc, linear_loss_fix_pct = linear_loss_results
        axs[0].plot(linear_loss_check_pct, linear_loss_acc, marker='*', color='y', label='linear_loss')
        axs[1].plot(linear_loss_check_pct, linear_loss_fix_pct, marker='*', color='y', label='linear_loss')
    if model_type == 'cb' and inf_k is not None:
        inf_label = 'all' if inf_k == -1 else 'topk_{}'.format(inf_k)
        influence_check_pct, influence_acc, influence_fix_pct = influence_results
        axs[0].plot(influence_check_pct, influence_acc, marker='+', color='m',
                    label='leaf_inf ({})'.format(inf_label))
        axs[1].plot(influence_check_pct, influence_fix_pct, marker='+', color='m',
                    label='leaf_inf ({})'.format(inf_label))
    axs[0].legend()
    axs[1].legend()

    if save_plot:
        plot_name = os.path.join(out_dir, dataset, 'cleaning.pdf')
        print('saving to {}...'.format(plot_name))
        os.makedirs(os.path.join(out_dir, dataset), exist_ok=True)
        plt.savefig(plot_name, format='pdf', bbox_inches='tight')

    plt.show()

    if save_results:
        efficiency_dir = os.path.join(out_dir, dataset, 'efficiency')
        effectiveness_dir = os.path.join(out_dir, dataset, 'effectiveness')
        os.makedirs(efficiency_dir, exist_ok=True)
        os.makedirs(effectiveness_dir, exist_ok=True)

        # save reference line
        np.save(os.path.join(effectiveness_dir, 'acc_test_clean.npy'), acc_test_clean)

        # ours
        np.save(os.path.join(efficiency_dir, 'our_{}_check_pct.npy'.format(settings)), our_check_pct)
        np.save(os.path.join(efficiency_dir, 'our_{}_fix_pct.npy'.format(settings)), our_fix_pct)
        np.save(os.path.join(effectiveness_dir, 'our_{}_check_pct.npy'.format(settings)), our_check_pct)
        np.save(os.path.join(effectiveness_dir, 'our_{}_acc.npy'.format(settings)), our_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='tree model to use.')
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