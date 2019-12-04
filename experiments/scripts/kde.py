"""
Explanation of an arbitrary test instance using kernel density estimation.
"""
import os
import sys
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner; TODO: remove this dependency

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
import seaborn as sns

from trex.explainer import TreeExplainer
from utility import model_util, data_util


def kde(model='lgb', encoding='leaf_output', dataset='nc17_mfc18', n_estimators=100,
        random_state=69, verbose=0, data_dir='data', pca_components=50, misclassified=False,
        save_results=False, out_dir='output/kde/', linear_model='svm', kernel='linear', true_label=True):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    explainer = TreeExplainer(tree, X_train, y_train, encoding=encoding, random_state=random_state,
                              dense_output=True, linear_model=linear_model, kernel=kernel,
                              use_predicted_labels=not true_label)

    # pick a test instance to explain
    test_preds = tree.predict(X_test)

    if misclassified:
        indices = np.where(y_test != test_preds)[0]
    else:
        indices = np.where(y_test == test_preds)[0]

    np.random.seed(random_state)
    test_ndx = np.random.choice(indices)
    x_test = X_test[[test_ndx]]
    test_pred = tree.predict(x_test)[0]
    test_pred2 = explainer.predict(x_test)[0]
    test_proba = tree.predict_proba(x_test)[0][1]

    # generate contributions to this prediction
    abs_contributions = np.abs(explainer.explain(x_test)[0])
    contributions = explainer.explain(x_test)[0]

    # split training data into pos and negative instances - based on true or predicted labels
    pos_label = test_pred
    if args.true_label:
        pos_indices = np.where(y_train == pos_label)[0]
        neg_indices = np.where(y_train != pos_label)[0]
    else:
        y_train_pred = tree.predict(X_train)
        pos_indices = np.where(y_train_pred == pos_label)[0]
        neg_indices = np.where(y_train_pred != pos_label)[0]

    print('\nExplaining test instance {}'.format(test_ndx))
    print('tree proba: {:.3f}, actual: {}'.format(test_proba, y_test[test_ndx]))

    if args.linear_model == 'lr':
        test_proba2 = explainer.predict_proba(x_test)[0][1]
        print('trex proba: {:.3f}'.format(test_proba2))

    if test_pred != test_pred2:
        print('tree and trex disagree!')

    pos_label_contribs = contributions[pos_indices]
    neg_label_contribs = contributions[neg_indices]

    abs_sort_ndx = np.argsort(abs_contributions)[::-1]
    pos_sort_ndx = np.argsort(np.abs(pos_label_contribs))[::-1]
    neg_sort_ndx = np.argsort(np.abs(neg_label_contribs))[::-1]

    abs_cumsum = np.cumsum(contributions[abs_sort_ndx])
    pos_cumsum = np.cumsum(pos_label_contribs[pos_sort_ndx])
    neg_cumsum = np.cumsum(neg_label_contribs[neg_sort_ndx])

    contribs_sum = np.sum(contributions)

    # trajectory when using KNN distances
    euclidean_sim = 1 / np.linalg.norm(X_train - x_test, axis=1)
    euclidean_sim_pos = euclidean_sim[pos_indices]
    euclidean_sim_neg = -euclidean_sim[neg_indices]
    euclidean_sim_sum = np.sum(euclidean_sim_pos) + np.sum(euclidean_sim_neg)

    euclid_pos_sort_ndx = np.argsort(np.abs(euclidean_sim_pos))[::-1]
    euclid_neg_sort_ndx = np.argsort(np.abs(euclidean_sim_neg))[::-1]

    euclid_cumsum_pos = np.cumsum(euclidean_sim_pos[euclid_pos_sort_ndx])
    euclid_cumsum_neg = np.cumsum(euclidean_sim_neg[euclid_neg_sort_ndx])

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(14, 4))

    ax0.axhline(euclidean_sim_sum, linestyle='--', color='k')
    ax0.plot(np.arange(len(euclid_cumsum_pos)), euclid_cumsum_pos, label='y={}'.format(pos_label), color='g')
    ax0.plot(np.arange(len(euclid_cumsum_neg)), euclid_cumsum_neg, label='y!={}'.format(pos_label), color='cyan')
    ax0.set_title('KNN')
    ax0.set_xlabel('# train instances')
    ax0.set_ylabel('1 / euclidean_dist')
    ax0.legend()

    ax1.axhline(contribs_sum, linestyle='--', color='k')
    ax1.plot(np.arange(len(pos_cumsum)), pos_cumsum, label='y={}'.format(pos_label), color='g')
    ax1.plot(np.arange(len(neg_cumsum)), neg_cumsum, label='y!={}'.format(pos_label), color='cyan')
    ax1.set_title('TREX')
    ax1.set_xlabel('# train instances')
    ax1.set_ylabel('contribution')
    ax1.legend()

    ax2.plot(np.arange(len(abs_cumsum)), abs_cumsum, color='purple')
    ax2.set_title('waterfall')
    ax2.set_xlabel('# train instances')
    ax2.set_ylabel('contribution')

    sns.kdeplot(pos_label_contribs, shade=True, label='y={}'.format(pos_label), ax=ax3, color='g')
    sns.kdeplot(neg_label_contribs, shade=True, label='y!={}'.format(pos_label), ax=ax3, color='cyan')
    ax3.set_title('kde')
    ax3.set_xlabel('contribution')
    ax3.set_ylabel('density')
    ax3.axvline(0)

    fig.suptitle('test{}, predicted as {}, actual {}'.format(test_ndx, test_pred, y_test[test_ndx]))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = os.path.join(out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, '{}_{}_{}_rs{}'.format(linear_model, kernel, encoding, random_state)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in the ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--pca_components', metavar='NUM', type=int, default=50, help='pca components.')
    parser.add_argument('--true_label', action='store_true', help='train TREX on the true labels.')
    parser.add_argument('--misclassified', action='store_true', help='explain misclassified test instance.')
    args = parser.parse_args()
    print(args)
    kde(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
        random_state=args.rs, linear_model=args.linear_model, kernel=args.kernel, misclassified=args.misclassified,
        pca_components=args.pca_components, true_label=args.true_label)
