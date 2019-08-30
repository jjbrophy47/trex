"""
Experiment: How well does the linear model approximate the tree ensemble?
"""
import os
import sys
import time
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner; TODO: remove this dependency

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import trex
from utility import model_util, data_util


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _plot_predictions(tree, explainer, data, ax=None, use_sigmoid=False):

    X_train, y_train, X_test, y_test = data
    multiclass = True if len(np.unique(y_train)) > 2 else False

    # tree ensemble predictions
    yhat_tree_train = tree.predict_proba(X_train)
    yhat_tree_test = tree.predict_proba(X_test)

    if not multiclass:
        yhat_tree_train = yhat_tree_train[:, 1].flatten()
        yhat_tree_test = yhat_tree_test[:, 1].flatten()
    else:
        yhat_tree_train = yhat_tree_train.flatten()
        yhat_tree_test = yhat_tree_test.flatten()

    # linear model predictions
    if explainer.linear_model == 'svm':
        yhat_linear_train = explainer.decision_function(X_train).flatten()
        yhat_linear_test = explainer.decision_function(X_test).flatten()
    else:
        yhat_linear_train = explainer.predict_proba(X_train)
        yhat_linear_test = explainer.predict_proba(X_test)

        if not multiclass:
            yhat_linear_train = yhat_linear_train[:, 1].flatten()
            yhat_linear_test = yhat_linear_test[:, 1].flatten()
        else:
            yhat_linear_train = yhat_linear_train.flatten()
            yhat_linear_test = yhat_linear_test.flatten()

    if use_sigmoid and explainer.linear_model == 'svm':
        yhat_linear_train = _sigmoid(yhat_linear_train)
        yhat_linear_test = _sigmoid(yhat_linear_test)

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

    res = {}
    res['tree'] = {'train': yhat_tree_train, 'test': yhat_tree_test}
    res['ours'] = {'train': yhat_linear_train, 'test': yhat_linear_test}
    return res


def fidelity(model='lgb', encoding='leaf_path', dataset='iris', n_estimators=100, random_state=69,
             true_label=False, data_dir='data', linear_model='svm', kernel='rbf', use_sigmoid=False,
             out_dir='output/fidelity/', flip_frac=None):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)
    data = X_train, y_train, X_test, y_test

    # corrupt the data if specified
    if flip_frac is not None:
        y_train, noisy_ndx = data_util.flip_labels(y_train, k=flip_frac, random_state=random_state)
        noisy_ndx = np.array(sorted(noisy_ndx))
        print('num noisy labels: {}'.format(len(noisy_ndx)))

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)

    # plot fidelity
    fig, ax = plt.subplots()
    start = time.time()
    explainer = trex.TreeExplainer(tree, X_train, y_train, encoding=encoding, linear_model=linear_model,
                                   kernel=kernel, random_state=random_state, use_predicted_labels=not true_label)
    results = _plot_predictions(tree, explainer, data, ax=ax, use_sigmoid=use_sigmoid)
    print('time: {:.3f}s'.format(time.time() - start))
    true_label_str = 'true_label' if true_label else ''
    sigmoid_str = 'sigmoid' if use_sigmoid else ''
    ax.set_xlabel('{}'.format(linear_model))
    ax.set_ylabel('{}'.format(model))
    ax.set_title('{}, {}\n{}, {}, {}, {}'.format(dataset, model, linear_model, kernel, encoding, true_label_str))
    ax.legend()
    plt.tight_layout()

    # save plot
    setting = '{}_{}_{}_{}_{}_{}'.format(model, linear_model, kernel, encoding, true_label_str, sigmoid_str)
    out_dir = os.path.join(out_dir, dataset, setting)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'fidelity.pdf'), format='pdf', bbox_inches='tight')

    # save data
    np.save(os.path.join(out_dir, 'tree_train.npy'), results['tree']['train'])
    np.save(os.path.join(out_dir, 'tree_test.npy'), results['tree']['test'])
    np.save(os.path.join(out_dir, 'ours_train.npy'), results['ours']['train'])
    np.save(os.path.join(out_dir, 'ours_test.npy'), results['ours']['test'])


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
    parser.add_argument('--use_sigmoid', action='store_true', default=False, help='Run svm results through sigmoid.')
    args = parser.parse_args()
    print(args)
    fidelity(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
             random_state=args.rs, true_label=args.true_label, linear_model=args.linear_model,
             kernel=args.kernel, use_sigmoid=args.use_sigmoid)
