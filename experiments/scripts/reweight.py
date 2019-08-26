"""
Experiment: Reweight the training instances that negatively contribute the most
towards the wrongly predicted label for misclassified test instances.
"""
import os
import sys
import argparse

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import sexee
from utility import model_util, data_util, exp_util


def _retrain(train_order, clf, X_train, y_train, X_test, y_test, k=6):
    """Return model performance as features are removed from the data."""

    auroc = []
    for i in tqdm.tqdm(range(1, k)):
        remove = train_order[:i]
        new_X_train = np.delete(X_train, remove, axis=1)
        new_X_test = np.delete(X_test, remove, axis=1)
        new_tree = clone(clf).fit(new_X_train, y_train)
        yhat = new_tree.predict_proba(new_X_test)[:, 1]
        auroc.append(roc_auc_score(y_test, yhat))

    return auroc


def _identify_instances(impact_sum, negative=False, threshold=0):
    """Idenitfies the support vectors that have a net positive / negative impact."""

    # get train instances that positively / negatively impact the predictions
    if negative:
        target_ndx = np.where(impact_sum < threshold)[0]
    else:
        target_ndx = np.where(impact_sum > threshold)[0]

    return target_ndx


def remove_data(model='lgb', dataset='nc17_mfc18', encoding='leaf_output', linear_model='lr', kernel='linear',
                n_estimators=100, random_state=69, test_size=0.2, iterations=1, n_points=150,
                data_dir='data', verbose=0, negative=False, threshold=0, absolute=False):

    remove_ndx = np.array([2, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 29, 31, 32, 33, 34])

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    X_train = np.delete(X_train, remove_ndx, axis=1)
    X_test = np.delete(X_test, remove_ndx, axis=1)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)
    original_auroc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding, kernel=kernel,
                                    random_state=random_state, linear_model=linear_model, dense_output=True)
    if verbose > 0:
        print(explainer)

    # get missed test instances
    missed_indices = np.where(tree.predict(X_test) != y_test)[0]
    correct_indices = np.where(tree.predict(X_test) == y_test)[0]
    print(missed_indices, missed_indices.shape)

    explain_ndx = np.where((y_test == 1) & (tree.predict(X_test) == 0))[0]
    model_util.performance(tree, X_train, y_train, X_test[missed_indices], y_test[missed_indices])

    # test_loss = exp_util.instance_loss(tree.predict_proba(X_test), y_test)
    # print(test_loss, test_loss.shape)
    # test_sort_ndx = np.argsort(test_loss)[::-1][:1000]

    # print(test_loss[test_sort_ndx])

    # compute total impact of train instances on test instances
    print('explaining...')
    contributions = explainer.explain(X_test[missed_indices])
    impact_sum = np.sum(contributions, axis=0)
    sort_ndx = np.argsort(impact_sum)
    impact_sum = impact_sum[sort_ndx]
    impact_sum = impact_sum / impact_sum.max()
    print(impact_sum, impact_sum.shape)
    print(len(np.where(impact_sum < 0)[0]))

    print('explaining...')
    contributions2 = explainer.explain(X_test[correct_indices])
    impact_sum2 = np.sum(contributions2, axis=0)
    sort_ndx2 = np.argsort(impact_sum2)[::-1]
    impact_sum2 = impact_sum2[sort_ndx2]
    impact_sum2 = impact_sum2 / impact_sum2.max()
    print(impact_sum2, impact_sum2.shape)
    print(len(np.where(impact_sum2 > 0)[0]))

    # get train instances that impact the predictions
    reweight_ndx = sort_ndx[:195]
    reweight_ndx2 = sort_ndx[:3244]
    # target_ndx = _identify_instances(impact_sum, negative=negative, threshold=threshold)
    # reweight_ndx = sort_ndx[target_ndx]
    # print(reweight_ndx, reweight_ndx.shape)

    sample_weight = np.ones(len(X_train))
    sample_weight[reweight_ndx] -= impact_sum[:195]
    sample_weight[reweight_ndx2] += impact_sum[:3244]

    tree = clone(clf).fit(X_train, y_train, sample_weight=sample_weight)
    model_util.performance(tree, X_train, y_train, X_test, y_test)
    model_util.performance(tree, X_train, y_train, X_test[missed_indices], y_test[missed_indices])

    # # remove offending train instances in segments and measure performance
    # auroc, n_removed = [], []

    # for i in range(0, 250 + 1, 25):

    #     # remove these instances from the train data
    #     delete_ndx = remove_ndx[:i]
    #     new_X_train = np.delete(X_train, delete_ndx, axis=0)
    #     new_y_train = np.delete(y_train, delete_ndx)

    #     tree = clone(clf).fit(new_X_train, new_y_train)
    #     auroc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
    #     n_removed.append(i)

    # print(n_removed, auroc)

    # fig, ax = plt.subplots()
    # ax.plot(n_removed, auroc, color='green', marker='.')
    # ax.axhline(original_auroc, linestyle='--', color='k')
    # ax.set_xlabel('train instances removed')
    # ax.set_ylabel('test auroc')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees for ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--iterations', metavar='NUM', type=int, default=1, help='Number of rounds.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set.')
    parser.add_argument('--negative', action='store_true', help='Remove / Relabel negative impact instances.')
    parser.add_argument('--absolute', action='store_true', help='Sort train instances by absolute value.')
    parser.add_argument('--n_points', default=50, help='Number of points to plot.')
    parser.add_argument('--verbose', default=0, help='Verbosity.')
    args = parser.parse_args()
    print(args)
    remove_data(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
                random_state=args.rs, test_size=args.test_size, iterations=args.iterations, negative=args.negative,
                absolute=args.absolute, n_points=args.n_points, linear_model=args.linear_model, kernel=args.kernel)
