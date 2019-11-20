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
from utility import model_util, data_util


def similarity(model='lgb', encoding='leaf_output', dataset='nc17_mfc18', n_estimators=100,
               random_state=69, verbose=0, data_dir='data', misclassified=False, alpha=1.0,
               save_results=False, out_dir='output/similarity/', linear_model='svm',
               kernel='linear', true_label=True):

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
    indices = np.where(y_test != test_preds)[0] if misclassified else np.where(y_test == test_preds)[0]

    np.random.seed(random_state)
    test_ndx = np.random.choice(indices)
    x_test = X_test[[test_ndx]]
    test_pred = tree.predict(x_test)[0]
    test_pred2 = explainer.predict(x_test)[0]
    test_proba = tree.predict_proba(x_test)[0][1]

    if test_pred != test_pred2:
        print('tree and trex disagree!')

    pos_label = test_pred

    # get training indices with the same label as the predicted label
    pos_indices = np.where(y_train == pos_label)[0]
    neg_indices = np.where(y_train != pos_label)[0]

    # get TREX similarity to this test instance
    trex_sim = explainer.similarity(x_test).flatten()
    trex_contribs = explainer.explain(x_test)[0]

    # get euclidean distance to this test instance
    euclidean_sim = np.linalg.norm(X_train - x_test, axis=1)

    print('test index: {}'.format(test_ndx))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    ax1.scatter(euclidean_sim[pos_indices], trex_sim[pos_indices], color='green', label='y={}'.format(pos_label),
                alpha=alpha)
    ax1.scatter(euclidean_sim[neg_indices], trex_sim[neg_indices], color='red', label='y!={}'.format(pos_label),
                alpha=alpha)
    ax1.set_xlabel('Euclidean distance')
    ax1.set_ylabel('TREX similarity')
    ax1.set_title('Similarity vs Distance')
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax1.legend()

    ax2.scatter(euclidean_sim[pos_indices], trex_contribs[pos_indices], color='green', label='y={}'.format(pos_label),
                alpha=alpha)
    ax2.scatter(euclidean_sim[neg_indices], trex_contribs[neg_indices], color='red', label='y!={}'.format(pos_label),
                alpha=alpha)
    ax2.set_xlabel('Euclidean distance')
    ax2.set_ylabel('TREX contribution')
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax2.set_title('Contributions vs Distance')
    ax2.legend()

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
    parser.add_argument('--alpha', type=float, default=1.0, help='transparency of the plots.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--true_label', action='store_true', help='train TREX on the true labels.')
    parser.add_argument('--misclassified', action='store_true', help='explain misclassified test instance.')
    args = parser.parse_args()
    print(args)
    similarity(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
               random_state=args.rs, linear_model=args.linear_model, kernel=args.kernel, alpha=args.alpha,
               misclassified=args.misclassified, true_label=args.true_label)
