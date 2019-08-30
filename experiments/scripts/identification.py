"""
Experiment: Idenitfy mismatched instances by training on NC17, generating predictions
    for MFC18, retraining on the predicted labels of MFC18, then use TREX
    to order the training instances, and see how efficient we are at finding
    the misclassified instances.
"""
import argparse
import os
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for trex

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import trex
from utility import model_util, data_util


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
    """Sorts train instances by largest weight."""

    train_weight = explainer.get_weight()[0]
    n_check = int(len(y_train) * cutoff_pct)

    train_order = np.argsort(np.abs(train_weight))[::-1][:n_check]
    interval = (n_check / len(y_train)) / points
    ckpt_ndx, fix_ndx = _record_fixes(train_order, noisy_ndx, len(y_train), interval)

    return ckpt_ndx, fix_ndx, interval, n_check, train_weight


def noise_detection(model_type='lgb', encoding='leaf_output', dataset='nc17_mfc18', linear_model='svm',
                    n_estimators=100, random_state=69, inf_k=None, data_dir='data',
                    true_label=False, kernel='linear', out_dir='output/cleaning',
                    save_plot=False, save_results=False, alpha=0.69, check_pct=0.3, values_pct=0.1):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # get model and data
    clf = model_util.get_classifier(model_type, n_estimators=n_estimators, random_state=random_state)
    X_pre_train, X_test, y_pre_train, y_test, label = data_util.get_data(dataset, random_state=random_state,
                                                                         data_dir=data_dir)

    remove_ndx = np.array([2, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 29, 31, 32, 33, 34])
    X_pre_train = np.delete(X_pre_train, remove_ndx, axis=1)
    X_test = np.delete(X_test, remove_ndx, axis=1)

    # split the test set into folds
    X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.5,
                                                        random_state=random_state, stratify=y_test)

    print('\npre-train instances: {}'.format(len(X_pre_train)))
    print('train instances: {}'.format(len(X_train)))
    print('test instances: {}\n'.format(len(X_test)))

    # train a tree ensemble on the training set
    tree = clone(clf).fit(X_pre_train, y_pre_train)
    model_util.performance(tree, X_pre_train, y_pre_train, X_train, y_train)

    # predict labels for part of the test set p1
    train_pred = tree.predict(X_train)

    # train a model p1 using the predicted labels
    model_noisy = clone(clf).fit(X_train, train_pred)
    model_util.performance(model_noisy, X_train, train_pred, X_test, y_test)

    # get indices of noisy labels (incorrectly predicted labels)
    noisy_ndx = np.where(train_pred != y_train)[0]
    noisy_ndx = np.array(sorted(noisy_ndx))
    print('\nnum noisy labels: {}'.format(len(noisy_ndx)))

    # train TREX on the noisy test set
    explainer = trex.TreeExplainer(model_noisy, X_train, train_pred, encoding=encoding, dense_output=True,
                                   random_state=random_state, use_predicted_labels=not true_label,
                                   kernel=kernel, linear_model=linear_model)

    # check accuracy before and after noise
    # acc_test_clean = accuracy_score(y_test, model_noisy.predict(X_test))
    # acc_test_noisy = accuracy_score(y_test, model_noisy.predict(X_test))

    acc_test_clean = roc_auc_score(y_test, model_noisy.predict_proba(X_test)[:, 1])
    acc_test_noisy = roc_auc_score(y_test, model_noisy.predict_proba(X_test)[:, 1])

    train_weight = explainer.get_weight()[0]
    train_order = np.argsort(np.abs(train_weight))[::-1]

    our_check_pct = []
    our_acc = []

    for pct in np.linspace(0, 0.3, 16):
        train_indices = train_order[:int(len(train_order) * pct)]
        new_y_train = data_util.flip_labels_with_indices(train_pred, train_indices)
        model_semi_noisy = clone(clf).fit(X_train, new_y_train)
        # acc_score = accuracy_score(y_test, model_semi_noisy.predict(X_test))
        acc_score = roc_auc_score(y_test, model_semi_noisy.predict_proba(X_test)[:, 1])
        print(acc_score)
        our_check_pct.append(pct)
        our_acc.append(acc_score)

    # ordering training instances
    # data = X_train, y_train, X_test, y_test
    # ckpt_ndx, fix_ndx, interval, n_check, our_train_weight = _our_method(explainer, noisy_ndx, y_train,
    #                                                                      cutoff_pct=check_pct)
    # our_results = _interval_performance(ckpt_ndx, fix_ndx, noisy_ndx, clf, data, acc_test_noisy)
    settings = '{}_{}'.format(linear_model, kernel)
    settings += '_true_label' if true_label else ''

    # plot results
    print('plotting...')
    # our_check_pct, our_acc, our_fix_pct = our_results

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(our_check_pct, our_acc, marker='.', color='g', label='ours')
    axs[0].axhline(acc_test_clean, color='k', linestyle='--')
    axs[0].set_xlabel('fraction of train data checked')
    axs[0].set_ylabel('test accuracy')
    # axs[1].plot(our_check_pct, our_fix_pct, marker='.', color='g', label='ours')
    # axs[1].set_xlabel('fraction of train data checked')
    # axs[1].set_ylabel('fraction of flips fixed')
    axs[0].legend()
    # axs[1].legend()

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
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--kernel', default='linear', help='Similarity kernel for the linear model.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--save_plot', action='store_true', default=False, help='Save plot results.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--check_pct', type=float, default=0.3, help='Max percentage of train instances to check.')
    parser.add_argument('--values_pct', type=float, default=0.1, help='Percentage of weights/loss values to compare.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    args = parser.parse_args()
    print(args)
    noise_detection(model_type=args.model, encoding=args.encoding, dataset=args.dataset, values_pct=args.values_pct,
                    linear_model=args.linear_model, kernel=args.kernel, check_pct=args.check_pct,
                    n_estimators=args.n_estimators, random_state=args.rs,
                    save_plot=args.save_plot,
                    save_results=args.save_results, true_label=args.true_label)
