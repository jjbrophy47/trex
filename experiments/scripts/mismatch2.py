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
from sklearn.metrics import accuracy_score, roc_auc_score

from util import model_util, data_util, exp_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble


def _display_stats(X, y, age_ndx=21, tag=''):
    """Show the distribution of labels for people in and not in the range 40-50."""

    n_age = len(np.where(X[:, age_ndx] == 1)[0])
    n_age_readmit = len(np.where((X[:, age_ndx] == 1) & (y == 1))[0])
    n_noage = len(np.where(X[:, age_ndx] != 1)[0])
    n_noage_readmit = len(np.where((X[:, age_ndx] != 1) & (y == 1))[0])
    n_readmit = len(np.where(y == 1)[0])
    print('{} {}/{} people aged 40-50 were readmitted.'.format(tag, n_age_readmit, n_age))
    print('{} {}/{} people NOT aged 40-50 were readmitted.'.format(tag, n_noage_readmit, n_noage))
    print('{} {}/{} people were readmitted'.format(tag, n_readmit, len(y)))


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


def _retrain(clf, tree, train_indices, test_indices, X_train, y_train, X_test, y_test):

    influence_scores = np.zeros((len(test_indices), len(train_indices)))

    for i, test_ndx in enumerate(tqdm.tqdm(test_indices)):
        x_test = X_test[test_ndx]
        ref_loss = np.abs(y_test[test_ndx] - tree.predict_proba(x_test)[1])

        for j, train_ndx in enumerate(train_indices):
            new_X_train = np.delete(X_train, train_ndx, axis=0)
            new_y_train = np.delete(y_train, train_ndx)
            new_tree = clone(clf).fit(new_X_train, new_y_train)
            query_loss = np.abs(y_test[test_ndx] - new_tree.predict_proba(x_test)[1])
            influence_scores[i][j] = query_loss - ref_loss

        print(influence_scores[i].mean())

    # shape=(n_test, n_train)
    return influence_scores


def mismatch(model='lgb', encoding='leaf_output', dataset='hospital2', n_estimators=100,
             random_state=69, plot=False, data_dir='data', age_ndx=21, sv_only=False,
             verbose=0, inf_k=None, n_subset=None, modify=True, aggregation='mean',
             train_true_label=False, impact_true_label=True, retrain=False, save_results=False,
             out_dir='output/mismatch/'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_feature=True)

    # train model on original data
    X_train, X_test, y_train, y_test, label, feature = data
    tree = clone(clf).fit(X_train, y_train)
    if verbose > 0:
        print('\nOriginal Data')
        _display_stats(X_train, y_train, tag='[Train]')
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    # introduce domain mismatch and train a model on the modified dataset
    if modify:
        X_train, y_train = _modify(X_train, y_train, age_ndx=age_ndx, random_state=random_state)
        tree = clone(clf).fit(X_train, y_train)
        if verbose > 0:
            print('\nModified Data')
            _display_stats(X_train, y_train, tag='[Train]')
            model_util.performance(tree, X_train, y_train, X_test, y_test)

    # get train instances for the 4 groups of interest
    train_age_readmit_ndx = np.where((X_train[:, age_ndx] == 1) & (y_train == 1))[0]
    train_age_noreadmit_ndx = np.where((X_train[:, age_ndx] == 1) & (y_train != 1))[0]
    train_noage_readmit_ndx = np.where((X_train[:, age_ndx] != 1) & (y_train == 1))[0]
    train_noage_noreadmit_ndx = np.where((X_train[:, age_ndx] != 1) & (y_train != 1))[0]

    # get test instances of interest
    test_age_ndx = np.where(X_test[:, age_ndx] == 1)[0]
    test_age_readmit_ndx = np.where((X_test[:, age_ndx] == 1) & (y_test == 1))[0]
    test_target_ndx = test_age_ndx

    # show stats and prformance for these test instances
    if verbose > 0:
        print('\nTarget test instances')
        _display_stats(X_test, y_test, tag='[Test]')
        test_age_acc = accuracy_score(y_test[test_age_ndx], tree.predict(X_test[test_age_ndx]))
        test_age_auroc = roc_auc_score(y_test[test_age_ndx], tree.predict_proba(X_test[test_age_ndx])[:, 1])
        print('[Test] people aged 40-50 accuracy: {:.3f}'.format(test_age_acc))
        print('[Test] people aged 40-50 auroc: {:.3f}'.format(test_age_auroc))

    # compute the most impactful train instances on the chosen test instances
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding, random_state=random_state,
                                    use_predicted_labels=not train_true_label)
    y = None if not impact_true_label else y_test[test_target_ndx]
    sv_ndx, sv_impact = explainer.train_impact(X_test[test_target_ndx], y=y)
    # sv_ndx, sv_impact = exp_util.sort_impact(sv_ndx, sv_impact)
    # sv_impact = np.array(sv_impact)

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
        print('no age, no readmit support vectors: {}'.format(len(sv_4_ndx)))
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

    if aggregation == 'mean':
        if sv_only:
            impact_1 = np.mean(sv_impact[sv_1_ndx], axis=0).mean()
            impact_2 = np.mean(sv_impact[sv_2_ndx], axis=0).mean()
            impact_3 = np.mean(sv_impact[sv_3_ndx], axis=0).mean()
            impact_4 = np.mean(sv_impact[sv_4_ndx], axis=0).mean()

        else:
            impact_1 = np.zeros((X_train.shape[0], sv_impact.shape[1]))
            impact_1[train_sv_1_ndx] = sv_impact[sv_1_ndx]
            impact_1 = impact_1[train_age_readmit_ndx]

            impact_2 = np.zeros((X_train.shape[0], sv_impact.shape[1]))
            impact_2[train_sv_2_ndx] = sv_impact[sv_2_ndx]
            impact_2 = impact_2[train_age_noreadmit_ndx]

            impact_3 = np.zeros((X_train.shape[0], sv_impact.shape[1]))
            impact_3[train_sv_3_ndx] = sv_impact[sv_3_ndx]
            impact_3 = impact_3[train_noage_readmit_ndx]

            impact_4 = np.zeros((X_train.shape[0], sv_impact.shape[1]))
            impact_4[train_sv_4_ndx] = sv_impact[sv_4_ndx]
            impact_4 = impact_4[train_noage_noreadmit_ndx]

    elif aggregation == 'sum':
        impact_1 = np.sum(sv_impact[sv_1_ndx])
        impact_2 = np.sum(sv_impact[sv_2_ndx])
        impact_3 = np.sum(sv_impact[sv_3_ndx])
        impact_4 = np.sum(sv_impact[sv_4_ndx])

    else:
        exit('{} aggregation unsupported'.format(aggregation))

    print('\nours:')
    print('age, readmit: {}'.format(impact_1.mean()))
    print('age, no readmit: {}'.format(impact_2.mean()))
    print('no age, readmit: {}'.format(impact_3.mean()))
    print('no age, no readmit: {}'.format(impact_4.mean()))

    if save_results:
        sexee_dir = os.path.join(out_dir, 'ours')
        os.makedirs(sexee_dir, exist_ok=True)
        np.save(os.path.join(sexee_dir, 'age_readmit.npy'), impact_1)
        np.save(os.path.join(out_dir, 'test_age_ndx.npy'), test_age_ndx)
        np.save(os.path.join(out_dir, 'test_age_readmit_ndx.npy'), test_age_readmit_ndx)

    if retrain:
        if sv_only:
            train_1_ndx = train_sv_1_ndx
            train_2_ndx = train_sv_2_ndx
            train_3_ndx = train_sv_3_ndx
            train_4_ndx = train_sv_4_ndx

        else:
            train_1_ndx = train_age_readmit_ndx
            train_2_ndx = train_age_noreadmit_ndx
            train_3_ndx = train_noage_readmit_ndx
            train_4_ndx = train_noage_noreadmit_ndx

        if save_results:
            retrain_dir = os.path.join(out_dir, 'retrain')
            os.makedirs(retrain_dir, exist_ok=True)

        start = time.time()
        age_readmit_scores = _retrain(clf, tree, train_1_ndx, test_target_ndx, X_train, y_train, X_test, y_test)
        print('age, readmit: {}, time: {:.3f}'.format(age_readmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(retrain_dir, 'age_readmit.npy'), age_readmit_scores)

        start = time.time()
        age_noreadmit_scores = _retrain(clf, tree, train_2_ndx, test_target_ndx, X_train, y_train, X_test, y_test)
        print('age, no readmit: {}, time: {:.3f}'.format(age_noreadmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(retrain_dir, 'age_noreadmit.npy'), age_noreadmit_scores)

        start = time.time()
        noage_readmit_scores = _retrain(clf, tree, train_3_ndx, test_target_ndx, X_train, y_train, X_test, y_test)
        print('no age, readmit: {}, time: {:.3f}'.format(noage_readmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(retrain_dir, 'noage_readmit.npy'), noage_readmit_scores)

        start = time.time()
        noage_noreadmit_scores = _retrain(clf, tree, train_4_ndx, test_target_ndx, X_train, y_train, X_test, y_test)
        print('no age, no readmit: {}, time: {:.3f}'.format(noage_noreadmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(retrain_dir, 'noage_noreadmit.npy'), noage_noreadmit_scores)

    # influence method
    if model == 'cb' and inf_k is not None:
        model_path = '.model.json'
        tree.save_model(model_path, format='json')

        if inf_k == -1:
            update_set = 'AllPoints'
        elif inf_k == 0:
            update_set = 'SinglePoint'
        else:
            update_set = 'TopKLeaves'

        if sv_only:
            train_1_ndx = train_sv_1_ndx
            train_2_ndx = train_sv_2_ndx
            train_3_ndx = train_sv_3_ndx
            train_4_ndx = train_sv_4_ndx

        else:
            train_1_ndx = train_age_readmit_ndx
            train_2_ndx = train_age_noreadmit_ndx
            train_3_ndx = train_noage_readmit_ndx
            train_4_ndx = train_noage_noreadmit_ndx

        if save_results:
            inf_dir = os.path.join(out_dir, 'influence')
            os.makedirs(inf_dir, exist_ok=True)

        print('\nleaf_influence:')
        leaf_influence = CBLeafInfluenceEnsemble(model_path, X_train, y_train, k=inf_k, update_set=update_set,
                                                 learning_rate=tree.learning_rate_)

        start = time.time()
        age_readmit_scores = _influence(leaf_influence, train_1_ndx, test_target_ndx, X_test, y_test)
        print('age, readmit: {}, time: {:.3f}'.format(age_readmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(inf_dir, 'age_readmit.npy'), age_readmit_scores)

        start = time.time()
        age_noreadmit_scores = _influence(leaf_influence, train_2_ndx, test_target_ndx, X_test, y_test)
        print('age, no readmit: {}, time: {:.3f}'.format(age_noreadmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(inf_dir, 'age_noreadmit.npy'), age_noreadmit_scores)

        start = time.time()
        noage_readmit_scores = _influence(leaf_influence, train_3_ndx, test_target_ndx, X_test, y_test)
        print('no age, readmit: {}, time: {:.3f}'.format(noage_readmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(inf_dir, 'noage_readmit.npy'), noage_readmit_scores)

        start = time.time()
        noage_noreadmit_scores = _influence(leaf_influence, train_4_ndx, test_target_ndx, X_test, y_test)
        print('no age, no readmit: {}, time: {:.3f}'.format(noage_noreadmit_scores.mean(), time.time() - start))

        if save_results:
            np.save(os.path.join(inf_dir, 'noage_noreadmit.npy'), noage_noreadmit_scores)


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
    parser.add_argument('--n_subset', type=int, default=None, help='Number of train instances to inspect.')
    parser.add_argument('--train_true_label', action='store_true', help='Train explainer on true labels.')
    parser.add_argument('--impact_true_label', action='store_true', help='Compute impact on true labels.')
    parser.add_argument('--modify', action='store_true', help='Modify the train data to skew the distribution.')
    parser.add_argument('--sv_only', action='store_true', help='Only use the support vectors when computing impact.')
    parser.add_argument('--retrain', action='store_true', help='Do leave-one-out retraining.')
    parser.add_argument('--aggregation', default='mean', help='Method to aggregate train impacts.')
    parser.add_argument('--save_results', action='store_true', help='Save the data from each method.')
    args = parser.parse_args()
    print(args)
    mismatch(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, inf_k=args.inf_k,
             verbose=args.verbose, n_subset=args.n_subset, aggregation=args.aggregation, modify=args.modify,
             sv_only=args.sv_only, train_true_label=args.train_true_label, impact_true_label=args.impact_true_label,
             retrain=args.retrain, save_results=args.save_results)
