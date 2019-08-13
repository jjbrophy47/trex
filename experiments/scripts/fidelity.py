"""
Experiment: How well does the linear model approximate the tree ensemble?
"""
import os
import sys
import time
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import sexee
from utility import model_util, data_util


def _plot_predictions(tree, explainer, data, ax=None):

    X_train, y_train, X_test, y_test = data

    # tree ensemble predictions
    yhat_tree_train = tree.predict_proba(X_train).flatten()
    yhat_tree_test = tree.predict_proba(X_test).flatten()

    # linear model predictions
    yhat_linear_train = explainer.decision_function(X_train).flatten()
    yhat_linear_test = explainer.decision_function(X_test).flatten()

    # compute correlation between tree probabilities and linear probabilities/decision values
    train_pear = np.corrcoef(yhat_tree_train, yhat_linear_train)[0][1]
    test_pear = np.corrcoef(yhat_tree_test, yhat_linear_test)[0][1]

    train_spear = spearmanr(yhat_tree_train, yhat_linear_train)[0]
    test_spear = spearmanr(yhat_tree_test, yhat_linear_test)[0]

    # plot results
    train_label = 'train={:.3f} (p), {:.3f} (s)'.format(train_pear, train_spear)
    test_label = 'test={:.3f} (p), {:.3f} (s)'.format(test_pear, test_spear)

    ax.scatter(yhat_linear_train, yhat_tree_train, color='blue', label=train_label)
    ax.scatter(yhat_linear_test, yhat_tree_test, color='cyan', label=test_label)


def fidelity(model='lgb', encoding='leaf_path', dataset='iris', n_estimators=100, random_state=69,
             true_label=False, data_dir='data', linear_model='svm', kernel='rbf', out_dir='output/fidelity/'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)
    data = X_train, y_train, X_test, y_test

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)

    # plot fidelity
    fig, ax = plt.subplots()
    start = time.time()
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding, linear_model=linear_model,
                                    kernel=kernel, random_state=random_state, use_predicted_labels=not true_label)
    _plot_predictions(tree, explainer, data, ax=ax)
    print('time: {:.3f}s'.format(time.time() - start))
    true_label_str = 'true_label' if true_label else ''
    ax.set_xlabel('svm decision')
    ax.set_ylabel('{} proba'.format(model))
    ax.set_title('{}, {}\n{}, {}, {}, {}'.format(dataset, model, linear_model, kernel, encoding, true_label_str))
    ax.legend()
    plt.tight_layout()

    # save plot
    out_dir = os.path.join(out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_fname = '{}_{}_{}_{}_{}.pdf'.format(model, linear_model, kernel, encoding, true_label_str)
    plt.savefig(os.path.join(out_dir, out_fname), format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='svm', help='linear model to use.')
    parser.add_argument('--kernel', type=str, default='linear', help='Similarity kernel.')
    parser.add_argument('--encoding', type=str, default='leaf_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--true_label', action='store_true', default=False, help='Use true labels for explainer.')
    args = parser.parse_args()
    print(args)
    fidelity(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
             random_state=args.rs, true_label=args.true_label, linear_model=args.linear_model, kernel=args.kernel)
