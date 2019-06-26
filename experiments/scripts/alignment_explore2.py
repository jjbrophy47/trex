"""
Exploration: Do an ablation test for each feature. See how removing each feature
affects test set performance.
"""
import argparse

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from util import model_util, data_util


def _remove_features(clf, X_train, y_train, X_test, y_test, plot=False):
    """Remove features one at a time, and measure their affect on the test set."""

    # train a tree ensemble
    tree = clone(clf).fit(X_train, y_train)
    base_test_auroc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    base_train_auroc = roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1])

    train_auroc = []
    test_auroc = []

    # train a new model after removing each feature
    for i in tqdm.tqdm(range(X_train.shape[1])):
        new_X_train = np.delete(X_train, i, axis=1)
        new_X_test = np.delete(X_test, i, axis=1)
        tree = clone(clf).fit(new_X_train, y_train)
        train_auroc.append(roc_auc_score(y_train, tree.predict_proba(new_X_train)[:, 1]))
        test_auroc.append(roc_auc_score(y_test, tree.predict_proba(new_X_test)[:, 1]))

    # save the feature that has the biggest impact
    max_ndx = np.argmax(test_auroc)

    # plot results
    if plot:
        fig, ax = plt.subplots()
        ax.axhline(base_test_auroc, color='k', linestyle='--')
        ax.scatter(np.arange(X_train.shape[1]), test_auroc, label='test')
        ax.set_ylabel('auroc')
        ax.set_xlabel('feature index')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return max_ndx, test_auroc[max_ndx], base_test_auroc, base_train_auroc


def ablation(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100, random_state=69,
             plot=False, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label, feature = data_util.get_data(dataset, random_state=random_state,
                                                                          data_dir=data_dir, return_feature=True)

    i = 0
    removed_index = []
    test_auroc = []
    original_test_auroc = None

    # keep greedily removing the most impactful feature while test set performance improves
    while True:
        results = _remove_features(clf, X_train, y_train, X_test, y_test, plot=plot)
        max_ndx, max_auroc, base_test_auroc, base_train_auroc = results
        og_ndx = max_ndx + i

        if i == 0:
            original_test_auroc = base_test_auroc

        if max_auroc <= base_test_auroc:
            break
        else:
            print('auroc before removal: {:.3f} (test), {:.3f} (train)'.format(base_test_auroc, base_train_auroc))
            print('auroc after removal: {:.3f} (test)'.format(max_auroc))
            print('removed feature: {}, index: {}, original index: {}'.format(feature[og_ndx], max_ndx, og_ndx))
            removed_index.append(og_ndx)
            test_auroc.append(max_auroc)
            X_train = np.delete(X_train, max_ndx, axis=1)
            X_test = np.delete(X_test, max_ndx, axis=1)

        i += 1

    print('features removed: {}'.format(removed_index))
    removed_feature = [feature[ndx][:10] for ndx in removed_index]
    removed_feature.insert(0, 'None')
    test_auroc.insert(0, original_test_auroc)

    # plot test auroc as max impact features are removed
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(removed_feature, test_auroc, label='test', marker='.', color='green')
    ax.axhline(original_test_auroc, color='k', linestyle='--')
    ax.set_ylabel('auroc')
    ax.set_xlabel('removed feature')
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
    ablation(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.plot)
