"""
Exploration: Compute the train set impact on an entire test set. See if this correlates with the
train instance support vector weights.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from scipy.stats import spearmanr

import sexee
from util import model_util, data_util


def _sort_impact(impact):
    """Returns a 1d array of impact_vals based on the train indices."""

    train_ndx, impact_vals = impact
    impact_vals = np.mean(impact_vals, axis=1)
    impact = zip(train_ndx, impact_vals)
    impact = sorted(impact, key=lambda tup: tup[0])
    _, impact_vals = zip(*impact)

    return impact_vals


def impact_vs_weight(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100, random_state=69,
                     plot=False, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding)

    # compute impact of train instances on test instances
    test_impact = explainer.train_impact(X_test)
    test_impact_vals = _sort_impact(test_impact)

    train_impact = explainer.train_impact(X_train)
    train_impact_vals = _sort_impact(train_impact)

    weight = explainer.get_train_weight()
    weight = sorted(weight, key=lambda tup: tup[0])
    _, weight = zip(*weight)

    # compute correlation between train impacts and train weights
    test_pear = np.corrcoef(test_impact_vals, weight)[0][1]
    train_pear = np.corrcoef(train_impact_vals, weight)[0][1]

    test_spear = spearmanr(test_impact_vals, weight)[0]
    train_spear = spearmanr(train_impact_vals, weight)[0]

    # plot results
    train_label = 'train={:.3f} (p), {:.3f} (s)'.format(train_pear, train_spear)
    test_label = 'test={:.3f} (p), {:.3f} (s)'.format(test_pear, test_spear)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(train_impact_vals, weight, marker='.', color='blue', label=train_label)
    ax.scatter(test_impact_vals, weight, marker='.', color='magenta', label=test_label)
    ax.set_ylabel('support vector coefficients')
    ax.set_xlabel('train instance impact')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medifor', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--plot', action='store_true', default=False, help='plots intermediate feaure removals.')
    args = parser.parse_args()
    print(args)
    impact_vs_weight(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.plot)
