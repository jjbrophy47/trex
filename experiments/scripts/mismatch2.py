"""
Experiment: Replicate domain mismatch experiment from Sharchilev et al.
"""
import os
import time
import argparse
import sys
sys.path.insert(0, os.getcwd())  # for influence_boosting
from copy import deepcopy

import tqdm
import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

from util import model_util, data_util, exp_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble


def _modify(X_train, y_train, age_ndx=21, remove_frac=0.9, random_state=69):
    """
    Do not have the same splits as Sharchilev, but will use a similar procedure.
    Remove percentage of readmitted people age 40-50 from training.
    """

    # Remove from the training set a fraction of people aged 40-50 who were readmitted
    target_ndx = np.where((y_train == 1) & (X_train[:, age_ndx] == 1))[0]
    np.random.seed(random_state)
    remove_ndx = np.random.choice(target_ndx, size=int(len(target_ndx) * remove_frac), replace=False)
    X_train_mod = np.delete(X_train, remove_ndx, axis=0)
    y_train_mod = np.delete(y_train, remove_ndx)

    n_orig_age = len(np.where(X_train[:, age_ndx] == 1)[0])
    n_orig_age_readmitted = len(np.where((X_train[:, age_ndx] == 1) & (y_train == 1))[0])
    n_orig_readmitted = len(np.where(y_train == 1)[0])
    print('[Train] Original data, {}/{} people aged 40-50 were readmitted.'.format(n_orig_age_readmitted, n_orig_age))
    print('[Train] Original data, {}/{} people were readmitted'.format(n_orig_readmitted, len(y_train)))

    n_mod_age = len(np.where(X_train_mod[:, age_ndx] == 1)[0])
    n_mod_age_readmitted = len(np.where((X_train_mod[:, age_ndx] == 1) & (y_train_mod == 1))[0])
    n_mod_readmitted = len(np.where(y_train_mod == 1)[0])
    print('[Train] Modified data, {}/{} people aged 40-50 were readmitted.'.format(n_mod_age_readmitted, n_mod_age))
    print('[Train] Modified data, {}/{} people were readmitted'.format(n_mod_readmitted, len(y_train_mod)))

    return X_train_mod, y_train_mod


def _influence(explainer, train_indices, test_indices, X_test, y_test):
    """
    Computes the influence of each train instance in X_train on all test instances in X_test.
    This uses the fastleafinfluence method by Sharchilev et al.
    """

    influence_scores = np.zeros((len(test_indices), len(train_indices)))
    buf = deepcopy(explainer)

    for i, test_ndx in enumerate(tqdm.tqdm(test_indices)):
        for j, train_ndx in enumerate(train_indices):
            explainer.fit(removed_point_idx=train_ndx, destination_model=buf)
            influence_scores[i][j] = buf.loss_derivative(X_test[[test_ndx]], y_test[[test_ndx]])[0]

    # shape=(n_test, n_train)
    return influence_scores


def mismatch(model='lgb', encoding='leaf_output', dataset='hospital2', n_estimators=100,
             random_state=69, plot=False, data_dir='data', age_ndx=21,
             verbose=0, inf_k=None, n_subset=50, true_label=False):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_feature=True)

    # train model on original data
    X_train, X_test, y_train, y_test, label, feature = data
    tree = clone(clf).fit(X_train, y_train)
    if verbose > 0:
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    # introduce domain mismatch and train a model on the modified dataset
    X_train_mod, y_train_mod = _modify(X_train, y_train, age_ndx=age_ndx, random_state=random_state)
    tree_mod = clone(clf).fit(X_train_mod, y_train_mod)
    if verbose > 0:
        model_util.performance(tree_mod, X_train, y_train, X_test, y_test)

    test_n_readmitted = len(np.where(y_test == 1)[0])
    test_age_ndx = np.where(X_test[:, age_ndx] == 1)[0]
    test_age_readmit_ndx = np.where((X_test[:, age_ndx] == 1) & (y_test == 1))[0]
    print('[Test] {}/{} people were readmitted'.format(test_n_readmitted, len(y_test)))
    print('[Test] {}/{} people aged 40-50 were readmitted'.format(len(test_age_readmit_ndx), len(test_age_ndx)))

    # get train instances for the 4 groups of interest
    train_age_readmit_ndx = np.where((X_train_mod[:, age_ndx] == 1) & (y_train_mod == 1))[0]
    train_age_noreadmit_ndx = np.where((X_train_mod[:, age_ndx] == 1) & (y_train_mod != 1))[0]
    train_noage_readmit_ndx = np.where((X_train_mod[:, age_ndx] != 1) & (y_train_mod == 1))[0]
    train_noage_noreadmit_ndx = np.where((X_train_mod[:, age_ndx] != 1) & (y_train_mod != 1))[0]

    # get test instances of interest
    test_target_ndx = test_age_ndx
    # test_age_noreadmitted_ndx = np.where((X_test[:, age_ndx] == 1) & (y_test != 1))[0]

    # compute the most impactful train instances on the chosen test instances
    explainer = sexee.TreeExplainer(tree_mod, X_train_mod, y_train_mod, encoding=encoding, random_state=random_state,
                                    use_predicted_labels=not true_label)
    sv_ndx, sv_impact = explainer.train_impact(X_test[test_target_ndx])
    sv_ndx, sv_impact = exp_util.sort_impact(sv_ndx, sv_impact)
    sv_impact = np.array(sv_impact)

    # filter out train instances that don't overlap with support vectors
    train_sv_1_ndx, _, sv_1_ndx = np.intersect1d(train_age_readmit_ndx, sv_ndx, return_indices=True)
    train_sv_2_ndx, _, sv_2_ndx = np.intersect1d(train_age_noreadmit_ndx, sv_ndx, return_indices=True)
    train_sv_3_ndx, _, sv_3_ndx = np.intersect1d(train_noage_readmit_ndx, sv_ndx, return_indices=True)
    train_sv_4_ndx, _, sv_4_ndx = np.intersect1d(train_noage_noreadmit_ndx, sv_ndx, return_indices=True)

    if verbose > 0:
        print('\ntotal support vectors: {}'.format(len(sv_ndx)))
        print('age, readmit support vectors: {}'.format(len(sv_1_ndx)))
        print('age, no readmit support vectors: {}'.format(len(sv_2_ndx)))
        print('no age, readmit support vectors: {}'.format(len(sv_3_ndx)))
        print('no age, no readmit support vectors: {}'.format(len(sv_3_ndx)))
        print('n_subset: {}'.format(n_subset))

    # choose only a subset of train instances in each group
    sv_1_ndx = sv_1_ndx[:n_subset]
    sv_2_ndx = sv_2_ndx[:n_subset]
    sv_3_ndx = sv_3_ndx[:n_subset]
    sv_4_ndx = sv_4_ndx[:n_subset]

    train_sv_1_ndx = train_sv_1_ndx[:n_subset]
    train_sv_2_ndx = train_sv_2_ndx[:n_subset]
    train_sv_3_ndx = train_sv_3_ndx[:n_subset]
    train_sv_4_ndx = train_sv_4_ndx[:n_subset]

    print('\nours:')
    print('age, readmit: {}'.format(sv_impact[sv_1_ndx].mean()))
    print('age, no readmit: {}'.format(sv_impact[sv_2_ndx].mean()))
    print('no age, readmit: {}'.format(sv_impact[sv_3_ndx].mean()))
    print('no age, no readmit: {}'.format(sv_impact[sv_4_ndx].mean()))

    # influence method
    if model == 'cb' and inf_k is not None:
        model_path = '.model.json'
        tree_mod.save_model(model_path, format='json')

        if inf_k == -1:
            update_set = 'AllPoints'
        elif inf_k == 0:
            update_set = 'SinglePoint'
        else:
            update_set = 'TopKLeaves'

        print('\nleaf_influence:')
        leaf_influence = CBLeafInfluenceEnsemble(model_path, X_train_mod, y_train_mod, k=inf_k, update_set=update_set,
                                                 learning_rate=tree_mod.learning_rate_)

        start = time.time()
        age_readmit_scores = _influence(leaf_influence, train_sv_1_ndx, test_target_ndx, X_test, y_test)
        age_readmit_mean = np.sum(age_readmit_scores, axis=0).mean()
        print('age, readmit: {}, time: {:.3f}'.format(age_readmit_mean, time.time() - start))

        start = time.time()
        age_noreadmit_scores = _influence(leaf_influence, train_sv_2_ndx, test_target_ndx, X_test, y_test)
        age_noreadmit_mean = np.sum(age_noreadmit_scores, axis=0).mean()
        print('age, no readmit: {}, time: {:.3f}'.format(age_noreadmit_mean, time.time() - start))

        start = time.time()
        noage_readmit_scores = _influence(leaf_influence, train_sv_3_ndx, test_target_ndx, X_test, y_test)
        noage_readmit_mean = np.sum(noage_readmit_scores, axis=0).mean()
        print('no age, readmit: {}, time: {:.3f}'.format(noage_readmit_mean, time.time() - start))

        start = time.time()
        noage_noreadmit_scores = _influence(leaf_influence, train_sv_4_ndx, test_target_ndx, X_test, y_test)
        noage_noreadmit_mean = np.sum(noage_noreadmit_scores, axis=0).mean()
        print('no age, no readmit: {}, time: {:.3f}'.format(noage_noreadmit_mean, time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='hospital2', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees for ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--verbose', type=int, default=0, help='Amount of output.')
    parser.add_argument('--n_subset', type=int, default=5, help='Number of train instances to inspect.')
    parser.add_argument('--true_label', action='store_true', help='Train explainer on true labels.')
    args = parser.parse_args()
    print(args)
    mismatch(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, true_label=args.true_label,
             inf_k=args.inf_k, verbose=args.verbose, n_subset=args.n_subset)
