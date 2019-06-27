"""
Experiment: Do the tree ensemble prediction probabilities correlate with the SVM decision values?
"""
import argparse

import sexee
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from util import model_util, data_util


# TODO: add another method to vary hyperparameters, and plot correlation as a function of the hyperparemeter
def prediction(model='lgb', encoding='tree_path', dataset='iris', n_estimators=100, random_state=69,
               timeit=False, true_labels=False, k=1000, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)
    n_classes = len(np.unique(y_train))

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)
    yhat_tree_train = tree.predict_proba(X_train)
    yhat_tree_test = tree.predict_proba(X_test)

    if n_classes > 2:
        yhat_tree_train = yhat_tree_train.flatten()
        yhat_tree_test = yhat_tree_test.flatten()
    else:
        yhat_tree_train = yhat_tree_train[:, 1]
        yhat_tree_test = yhat_tree_test[:, 1]

    # train an svm on the tree ensemble features
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding, random_state=random_state,
                                    timeit=timeit, use_predicted_labels=not true_labels)
    model_util.performance(explainer, X_train, y_train, X_test, y_test)
    train_feature = explainer.extractor_.transform(X_train)
    test_feature = explainer.extractor_.transform(X_test)

    # get svm decision outputs
    svm = explainer.get_svm()
    yhat_svm_train = svm.decision_function(train_feature).flatten()
    yhat_svm_test = svm.decision_function(test_feature).flatten()

    # compute correlation between tree probabilities and svm decision values
    train_pear = np.corrcoef(yhat_tree_train, yhat_svm_train)[0][1]
    test_pear = np.corrcoef(yhat_tree_test, yhat_svm_test)[0][1]

    train_spear = spearmanr(yhat_tree_train, yhat_svm_train)[0]
    test_spear = spearmanr(yhat_tree_test, yhat_svm_test)[0]

    # plot results
    train_label = 'train={:.3f} (p), {:.3f} (s)'.format(train_pear, train_spear)
    test_label = 'test={:.3f} (p), {:.3f} (s)'.format(test_pear, test_spear)

    fig, ax = plt.subplots()
    ax.scatter(yhat_svm_train[:k], yhat_tree_train[:k], color='blue', label=train_label)
    ax.scatter(yhat_svm_test[:k], yhat_tree_test[:k], color='cyan', label=test_label)
    ax.set_xlabel('svm decision')
    ax.set_ylabel('{} proba'.format(model))
    ax.set_title('Fidelity ({}, {}, {})'.format(model, encoding, dataset))
    ax.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    parser.add_argument('--true_labels', action='store_true', default=False, help='Use true labels for explainer.')
    args = parser.parse_args()
    print(args)
    prediction(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.timeit, args.true_labels)
