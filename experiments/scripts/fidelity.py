"""
Experiment: Tests the effect that different feature representations and SVM hyperparameters
have on the fidelity between the SVM and the tree ensemble.
"""
import argparse

import tqdm
import sexee
import matplotlib.pyplot as plt

from util import model_util, data_util


def _get_values(variable='C'):

    if variable == 'C':
        # result = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
        result = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    else:
        exit('{} not recorgnized'.format(variable))

    return result


def _get_explainer(model, X_train, y_train, encoding, random_state, val, variable='C'):

    if variable == 'C':
        result = sexee.TreeExplainer(model, X_train, y_train, encoding=encoding, random_state=random_state, C=val)
    else:
        exit('{} not recorgnized'.format(variable))

    return result


def fidelity(variable='C', model='lgb', encoding='tree_path', dataset='iris', n_estimators=100,
             random_state=69, timeit=False):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state)

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)
    yhat_tree_train = tree.predict(X_train)
    yhat_tree_test = tree.predict(X_test)

    # get values to test over
    vals = _get_values(variable=variable)

    # compute change in fidelity by changing the hyperparameter
    train_overlaps = []
    test_overlaps = []

    for val in tqdm.tqdm(vals):
        exp = _get_explainer(tree, X_train, y_train, encoding, random_state, val, variable=variable)
        train_feature = exp.extractor_.transform(X_train)
        test_feature = exp.extractor_.transform(X_test)

        svm = exp.get_svm()
        yhat_svm_train = svm.predict(train_feature)
        yhat_svm_test = svm.predict(test_feature)

        train_overlaps.append(model_util.fidelity(yhat_tree_train, yhat_svm_train))
        test_overlaps.append(model_util.fidelity(yhat_tree_test, yhat_svm_test))

    # convert overlaps to percentages
    train_overlap_pct = [len(overlaps) / len(y_train) for overlaps in train_overlaps]
    test_overlap_pct = [len(overlaps) / len(y_test) for overlaps in test_overlaps]

    print(train_overlap_pct)
    print(test_overlap_pct)

    fig, ax = plt.subplots()
    ax.plot(vals, train_overlap_pct, marker='.', label='train')
    ax.plot(vals, test_overlap_pct, marker='.', label='test')
    ax.set_xscale('log')
    ax.set_xlabel(variable)
    ax.set_ylabel('tree-svm prediction overlap (%)')
    ax.set_title('SVM Fidelity ({}, {}, {})'.format(model, encoding, dataset))
    ax.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--variable', type=str, default='C', help='hyperameter to analyze.')
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    args = parser.parse_args()
    print(args)
    fidelity(args.variable, args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.timeit)
