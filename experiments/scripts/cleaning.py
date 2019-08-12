"""
Experiment: dataset cleaning from noisy labels, specifically a percentage of flipped labels
in the train data. Only for binary classification datasets.
"""
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

import sexee
from utility import model_util, data_util, exp_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble


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
    """Sorts train instances by largest weight support vectors."""

    train_weight = explainer.get_weight()[0]
    n_check = len(np.where(np.abs(train_weight) > 0)[0])

    if n_check == len(y_train):
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

    # sort by descending order; train instances that increase the log loss the most are first
    train_order = np.argsort(influence_scores)[::-1][:n_check]
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, n_train, interval)
    return ckpt_ndx, fix_ndx, influence_scores, train_order


def noise_detection(model_type='lgb', encoding='tree_path', dataset='iris', linear_model='svm',
                    n_estimators=100, random_state=69, linear_model_loss=False, inf_k=None, data_dir='data',
                    flip_frac=0.4, true_label=False, kernel='rbf', out_dir='output/cleaning',
                    save_plot=False, save_results=False, alpha=0.69, check_pct=0.3, values_pct=0.1):
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

    # our method
    print('\nordering by our method...')
    explainer = sexee.TreeExplainer(model_noisy, X_train, y_train_noisy, encoding=encoding, dense_output=True,
                                    random_state=random_state, use_predicted_labels=not true_label,
                                    kernel=kernel, linear_model=linear_model)
    ckpt_ndx, fix_ndx, interval, n_check, our_train_weight = _our_method(explainer, noisy_ndx, y_train,
                                                                         cutoff_pct=check_pct)
    our_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # random method
    print('ordering by random...')
    ckpt_ndx, fix_ndx = _random_method(noisy_ndx, y_train, interval, to_check=n_check, random_state=random_state)
    random_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # tree loss method
    print('ordering by tree loss...')
    y_train_proba = model_noisy.predict_proba(X_train)
    ckpt_ndx, fix_ndx, tree_loss, tree_ndx = _loss_method(noisy_ndx, y_train_proba, y_train_noisy, interval,
                                                          to_check=n_check)
    tree_loss_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

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
        ckpt_ndx, fix_ndx, linear_loss, linear_ndx = _loss_method(noisy_ndx, y_train_proba, y_train, interval,
                                                                  to_check=n_check)
        linear_loss_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

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
                                                 learning_rate=model.learning_rate_, update_set=update_set, k=inf_k)
        ckpt_ndx, fix_ndx, inf_scores, inf_order = _influence_method(leaf_influence, noisy_ndx, X_train, y_train,
                                                                     y_train_noisy, interval, to_check=n_check)
        influence_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)

    # plot training weight/loss values
    instance_vals_dir = 'output/cleaning/{}/instance_values/'.format(dataset)
    os.makedirs(instance_vals_dir, exist_ok=True)

    # ours vs tree loss
    values_pct_str = '{}%'.format(int(values_pct * 100))
    values_cutoff = int(len(y_train) * values_pct)
    top_our_ndx = np.argsort(np.abs(our_train_weight))[::-1][:values_cutoff]
    top_tree_ndx = np.argsort(np.abs(tree_loss))[::-1][:values_cutoff]

    fig, ax = plt.subplots()
    ax.scatter(np.abs(our_train_weight[top_our_ndx]), tree_loss[top_our_ndx], color='green',
               label='top {}: ours'.format(values_pct_str), alpha=alpha, marker='o')
    ax.scatter(np.abs(our_train_weight[top_tree_ndx]), tree_loss[top_tree_ndx], color='orange',
               label='top {}: tree_loss'.format(values_pct_str), alpha=alpha, marker='X')
    ax.set_ylabel('tree-ensemble (L1 loss)')
    ax.set_xlabel(r'ours (|$\alpha_i$|)')
    ax.legend()
    plt.savefig(os.path.join(instance_vals_dir, 'tree_loss.pdf'), format='pdf', bbox_inches='tight')
    np.save(os.path.join(instance_vals_dir, 'our{}_instance_vals.npy'.format(linear_model)), our_train_weight)
    np.save(os.path.join(instance_vals_dir, 'tree_instance_vals.npy'), tree_loss)

    # ours vs linear loss
    if linear_model_loss:
        top_linear_ndx = np.argsort(np.abs(linear_loss))[::-1][:values_cutoff]

        fig, ax = plt.subplots()
        ax.scatter(np.abs(our_train_weight[top_our_ndx]), linear_loss[top_our_ndx], color='green',
                   label='top {}: ours'.format(values_pct_str), alpha=alpha, marker='o')
        ax.scatter(np.abs(our_train_weight[top_linear_ndx]), linear_loss[top_linear_ndx], color='orange',
                   label='top {}: {}_loss'.format(values_pct_str, linear_model), alpha=alpha, marker='X')
        ax.set_ylabel('{} (L1 loss)'.format(linear_model))
        ax.set_xlabel(r'ours (|$\alpha_i$|)')
        ax.legend()
        plt.savefig(os.path.join(instance_vals_dir, 'linear_loss.pdf'), format='pdf', bbox_inches='tight')
        np.save(os.path.join(instance_vals_dir, 'linear{}_instance_vals.npy'.format(linear_model)), linear_loss)

    # ours vs leafinfluence
    if model_type == 'cb' and inf_k is not None:
        top_inf_ndx = np.argsort(inf_scores)[::-1][:values_cutoff]

        fig, ax = plt.subplots()
        ax.scatter(np.abs(our_train_weight[top_our_ndx]), inf_scores[top_our_ndx], color='green',
                   label='top {}: ours'.format(values_pct_str), alpha=alpha, marker='o')
        ax.scatter(np.abs(our_train_weight[top_inf_ndx]), inf_scores[top_inf_ndx], color='orange',
                   label='top {}: leaf_inf'.format(values_pct_str), alpha=alpha, marker='X')
        ax.set_ylabel('leafinfluence')
        ax.set_xlabel(r'ours (|$\alpha_i$|)')
        ax.legend()
        plt.savefig(os.path.join(instance_vals_dir, 'influence.pdf'), format='pdf', bbox_inches='tight')
        np.save(os.path.join(instance_vals_dir, 'influence_instance_vals.npy'), inf_scores)

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
    axs[0].set_ylabel('test accuracy')
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
        np.save(os.path.join(efficiency_dir, 'our{}_check_pct.npy'.format(linear_model)), our_check_pct)
        np.save(os.path.join(efficiency_dir, 'our{}_fix_pct.npy'.format(linear_model)), our_fix_pct)
        np.save(os.path.join(effectiveness_dir, 'our{}_check_pct.npy'.format(linear_model)), our_check_pct)
        np.save(os.path.join(effectiveness_dir, 'our_{}acc.npy'.format(linear_model)), our_acc)

        # random
        np.save(os.path.join(efficiency_dir, 'rand_check_pct.npy'), rand_check_pct)
        np.save(os.path.join(efficiency_dir, 'rand_fix_pct.npy'), rand_fix_pct)
        np.save(os.path.join(effectiveness_dir, 'rand_check_pct.npy'), rand_check_pct)
        np.save(os.path.join(effectiveness_dir, 'rand_acc.npy'), rand_acc)

        # tree loss
        np.save(os.path.join(efficiency_dir, 'tree_loss_check_pct.npy'), tree_loss_check_pct)
        np.save(os.path.join(efficiency_dir, 'tree_loss_fix_pct.npy'), tree_loss_fix_pct)
        np.save(os.path.join(effectiveness_dir, 'tree_loss_check_pct.npy'), tree_loss_check_pct)
        np.save(os.path.join(effectiveness_dir, 'tree_loss_acc.npy'), tree_loss_acc)

        if linear_model_loss:
            np.save(os.path.join(efficiency_dir, 'linear{}_loss_check_pct.npy'.format(linear_model)),
                    linear_loss_check_pct)
            np.save(os.path.join(efficiency_dir, 'linear{}_loss_fix_pct.npy'.format(linear_model)),
                    linear_loss_fix_pct)
            np.save(os.path.join(effectiveness_dir, 'linear{}_loss_check_pct.npy'.format(linear_model)),
                    linear_loss_check_pct)
            np.save(os.path.join(effectiveness_dir, 'linear{}_loss_acc.npy'.format(linear_model)),
                    linear_loss_acc)

        if model_type == 'cb' and inf_k is not None:
            np.save(os.path.join(efficiency_dir, 'influence_check_pct.npy'), influence_check_pct)
            np.save(os.path.join(efficiency_dir, 'influence_fix_pct.npy'), influence_fix_pct)
            np.save(os.path.join(effectiveness_dir, 'influence_check_pct.npy'), influence_check_pct)
            np.save(os.path.join(effectiveness_dir, 'influence_acc.npy'), influence_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='tree model to use.')
    parser.add_argument('--linear_model', type=str, default='svm', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--kernel', default='rbf', help='Similarity kernel for the linear model.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--linear_model_loss', action='store_true', default=False, help='Include linear loss.')
    parser.add_argument('--save_plot', action='store_true', default=False, help='Save plot results.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--flip_frac', type=float, default=0.4, help='Fraction of train labels to flip.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--check_pct', type=float, default=0.3, help='Max percentage of train instances to check.')
    parser.add_argument('--values_pct', type=float, default=0.1, help='Percentage of weights/loss values to compare.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    args = parser.parse_args()
    print(args)
    noise_detection(model_type=args.model, encoding=args.encoding, dataset=args.dataset, values_pct=args.values_pct,
                    linear_model=args.linear_model, kernel=args.kernel, check_pct=args.check_pct,
                    n_estimators=args.n_estimators, random_state=args.rs, linear_model_loss=args.linear_model_loss,
                    save_plot=args.save_plot, flip_frac=args.flip_frac, inf_k=args.inf_k,
                    save_results=args.save_results, true_label=args.true_label)
