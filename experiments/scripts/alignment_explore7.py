"""
Exploration: Identify the most impactful train instances for a specific group of test instances.
    Specifically for test instances that are in some way similar to each other, such as they have the same
    manipulation types, or some other domain knowledge. Examine the identified train instances.
"""
import argparse

import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

from util import model_util, data_util


def _sort_impact(sv_ndx, impact):
    """Sorts support vectors by absolute impact values."""

    if impact.ndim == 2:
        impact = np.sum(impact, axis=1)
    impact_list = zip(sv_ndx, impact)
    impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)

    sv_ndx, impact = zip(*impact_list)
    sv_ndx = np.array(sv_ndx)
    return sv_ndx, impact


def manipulations(model='lgb', encoding='tree_path', dataset='NC17_EvalPart1', n_estimators=100, random_state=69,
                  topk_train=5, test_subset=50, test_size=0.1, show_performance=True, true_label=False,
                  data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_manipulation=True,
                              test_size=test_size)
    X_train, X_test, y_train, y_test, label, manip_train, manip_test, manip_label = data

    # fit a tree ensemble and an explainer for that tree ensemble
    tree = clone(clf).fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding, use_predicted_labels=not true_label)

    if show_performance:
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    # pick a subset of test instances to explain
    manip_type = 'pastesplice'
    manip_col = np.where(manip_label == manip_type)[0][0]
    y_manip_test = manip_test[:, manip_col]
    y_manip_train = manip_train[:, manip_col]

    manip_train_sum = np.sum(manip_train, axis=1)
    manip_test_sum = np.sum(manip_test, axis=1)
    single_test_manip = np.where(manip_test_sum == 1)[0]
    single_train_manip = np.where(manip_train_sum == 1)[0]
    # print(single_manip)
    # print(manip_sum, manip_sum.shape)

    print('{} in train: {}'.format(manip_type, np.sum(y_manip_train)))
    print('{} in test: {}'.format(manip_type, np.sum(y_manip_test)))

    manip_train_ndx = np.where(y_manip_train == 1)[0]
    manip_test_ndx = np.where(y_manip_test == 1)[0]

    manip_train_ndx = np.array(list(set(manip_test_ndx).intersection(set(single_test_manip))))
    manip_test_ndx = np.array(list(set(manip_train_ndx).intersection(set(single_train_manip))))

    print(manip_train_ndx, len(manip_train_ndx))
    print(manip_test_ndx, len(manip_test_ndx))

    # n_test_sample = min(np.sum(y_manip_test), test_subset)
    n_test_sample = len(manip_test_ndx)
    print('sampling {} in test with {}'.format(n_test_sample, manip_type))
    np.random.seed(random_state)
    manip_test_ndx = np.random.choice(manip_test_ndx, size=n_test_sample, replace=False)
    X_test_manip = X_test[manip_test_ndx]

    sv_ndx, impact = explainer.train_impact(X_test_manip)
    sv_ndx, impact = _sort_impact(sv_ndx, impact)

    impact_list = zip(sv_ndx, impact)
    pos_impact_list = [impact_item for impact_item in impact_list if impact_item[1] > 0]
    pos_sv_ndx, pos_impact = zip(*pos_impact_list)
    topk_pos_sv_ndx = pos_sv_ndx[:topk_train]

    manip_train_overlap = set(manip_train_ndx).intersection(set(sv_ndx))
    print('{} in support vectors: {}'.format(manip_type, len(manip_train_overlap)))

    overlap = set(topk_pos_sv_ndx).intersection(set(manip_train_ndx))
    print('{} in top {} support vectors: {}'.format(manip_type, topk_train, len(overlap)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='NC17_EvalPart1', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', default=5, type=int, help='train subset to use.')
    parser.add_argument('--test_subset', default=50, type=int, help='test subset to use.')
    parser.add_argument('--test_size', default=0.2, type=float, help='size of the test set.')
    args = parser.parse_args()
    print(args)
    manipulations(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.topk_train,
                  args.test_subset, args.test_size)
