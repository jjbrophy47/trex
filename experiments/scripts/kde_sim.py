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

from trex.explainer import TreeExplainer
from utility import model_util, data_util, print_util


def kde_sim(args, model='lgb', encoding='leaf_output', dataset='nc17_mfc18', n_estimators=100,
            random_state=69, verbose=0, data_dir='data', misclassified=False,
            save_results=False, out_dir='output/kde_sim/', linear_model='svm', kernel='linear', true_label=True):

    # make logger
    setting = '{}_{}_{}_rs{}'.format(linear_model, kernel, encoding, random_state)
    out_dir = os.path.join(out_dir, dataset)
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(log_dir, '{}.txt'.format(setting)))
    logger.info(args)

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test, logger=logger)

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

    logger.info('\nExplaining test instance {}'.format(test_ndx))
    logger.info('tree proba: {:.3f}, actual: {}'.format(test_proba, y_test[test_ndx]))

    if args.linear_model == 'lr':
        test_proba2 = explainer.predict_proba(x_test)[0][1]
        logger.info('trex proba: {:.3f}'.format(test_proba2))

    if test_pred != test_pred2:
        logger.info('tree and trex disagree!')

    pos_label_contribs = contributions[pos_indices]
    neg_label_contribs = contributions[neg_indices]

    pos_sort_ndx = np.argsort(np.abs(pos_label_contribs))[::-1]
    neg_sort_ndx = np.argsort(np.abs(neg_label_contribs))[::-1]

    pos_cumsum = np.cumsum(pos_label_contribs[pos_sort_ndx])
    neg_cumsum = np.cumsum(neg_label_contribs[neg_sort_ndx])

    contribs_sum = np.sum(contributions)

    # trajectory when using KNN distances
    knn_sim = 1 / np.linalg.norm(X_train - x_test, axis=1)
    knn_sim_pos = knn_sim[pos_indices]
    knn_sim_neg = -knn_sim[neg_indices]
    knn_sim_sum = np.sum(knn_sim_pos) + np.sum(knn_sim_neg)

    knn_pos_sort_ndx = np.argsort(np.abs(knn_sim_pos))[::-1]
    knn_neg_sort_ndx = np.argsort(np.abs(knn_sim_neg))[::-1]

    knn_cumsum_pos = np.cumsum(knn_sim_pos[knn_pos_sort_ndx])
    knn_cumsum_neg = np.cumsum(knn_sim_neg[knn_neg_sort_ndx])

    # trajectory when using TREX-KNN distances
    X_train_alt = explainer.transform(X_train)
    x_test_alt = explainer.transform(x_test)
    trex_knn_sim = 1 / np.linalg.norm(X_train_alt - x_test_alt, axis=1)
    trex_knn_sim_pos = trex_knn_sim[pos_indices]
    trex_knn_sim_neg = -trex_knn_sim[neg_indices]
    trex_knn_sim_sum = np.sum(trex_knn_sim_pos) + np.sum(trex_knn_sim_neg)

    trex_knn_pos_sort_ndx = np.argsort(np.abs(trex_knn_sim_pos))[::-1]
    trex_knn_neg_sort_ndx = np.argsort(np.abs(trex_knn_sim_neg))[::-1]

    trex_knn_cumsum_pos = np.cumsum(trex_knn_sim_pos[trex_knn_pos_sort_ndx])
    trex_knn_cumsum_neg = np.cumsum(trex_knn_sim_neg[trex_knn_neg_sort_ndx])

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    ax = axs[0][0]
    ax.axhline(contribs_sum, linestyle='--', color='k')
    ax.plot(np.arange(len(pos_cumsum)), pos_cumsum, label='y={}'.format(pos_label), color='g')
    ax.plot(np.arange(len(neg_cumsum)), neg_cumsum, label='y!={}'.format(pos_label), color='cyan')
    ax.set_title('TREX-{}'.format(linear_model))
    ax.set_xlabel('# train instances')
    ax.set_ylabel('contribution')
    ax.legend()

    ax = axs[0][1]
    ax.axhline(knn_sim_sum, linestyle='--', color='k')
    ax.plot(np.arange(len(knn_cumsum_pos)), knn_cumsum_pos, label='y={}'.format(pos_label), color='g')
    ax.plot(np.arange(len(knn_cumsum_neg)), knn_cumsum_neg, label='y!={}'.format(pos_label), color='cyan')
    ax.set_title('KNN')
    ax.set_xlabel('# train instances')
    ax.set_ylabel('1 / euclidean_dist')
    ax.legend()

    ax = axs[0][2]
    ax.axhline(trex_knn_sim_sum, linestyle='--', color='k')
    ax.plot(np.arange(len(trex_knn_cumsum_pos)), trex_knn_cumsum_pos, label='y={}'.format(pos_label), color='g')
    ax.plot(np.arange(len(trex_knn_cumsum_neg)), trex_knn_cumsum_neg, label='y!={}'.format(pos_label), color='cyan')
    ax.set_title('TREX-KNN')
    ax.set_xlabel('# train instances')
    ax.set_ylabel('1 / euclidean_dist')
    ax.legend()

    fig.suptitle('test{}, predicted as {}, actual {}'.format(test_ndx, test_pred, y_test[test_ndx]))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # comparison plots
    trex_contribs = explainer.explain(x_test)[0]

    # KNN: original feature space
    knn_dist = np.linalg.norm(X_train - x_test, axis=1)
    trex_knn_dist = np.linalg.norm(X_train_alt - x_test_alt, axis=1)

    ax = axs[1][0]
    ax.scatter(knn_dist[pos_indices], trex_contribs[pos_indices], color='green', label='y={}'.format(pos_label))
    ax.scatter(knn_dist[neg_indices], trex_contribs[neg_indices], color='red', label='y!={}'.format(pos_label))
    ax.set_xlabel('Euclidean distance')
    ax.set_ylabel('TREX contribution')
    ax.set_title('TREX-{} vs KNN'.format(linear_model))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.legend()

    ax = axs[1][1]
    ax.scatter(knn_dist[pos_indices], trex_contribs[pos_indices], color='green', label='y={}'.format(pos_label))
    ax.scatter(knn_dist[neg_indices], trex_contribs[neg_indices], color='red', label='y!={}'.format(pos_label))
    ax.set_xlabel('Euclidean distance')
    ax.set_ylabel('TREX contribution')
    ax.set_title('TREX-{} vs KNN (Zoomed in)'.format(linear_model))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.set_xlim(0, 200000)
    ax.legend()

    ax = axs[1][2]
    ax.scatter(trex_knn_dist[pos_indices], trex_contribs[pos_indices], color='green', label='y={}'.format(pos_label))
    ax.scatter(trex_knn_dist[neg_indices], trex_contribs[neg_indices], color='red', label='y!={}'.format(pos_label))
    ax.set_xlabel('Euclidean distance')
    ax.set_ylabel('TREX contribution')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.set_title('TREX-{} vs TREX-KNN'.format(linear_model))
    ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(wspace=0.25, hspace=0.35)
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
    parser.add_argument('--true_label', action='store_true', help='train TREX on the true labels.')
    parser.add_argument('--misclassified', action='store_true', help='explain misclassified test instance.')
    args = parser.parse_args()
    kde_sim(args, model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
            random_state=args.rs, linear_model=args.linear_model, kernel=args.kernel, misclassified=args.misclassified,
            true_label=args.true_label)
