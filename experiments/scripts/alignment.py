"""
Experiment: Test to see if we can improve performance to the MFC18_EvalPart1 test set by re-weighting or dropping
training data from NC17_EvalPart1; or if we can identify instances to be added to the train data.
"""
import argparse

import shap
import tqdm
import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from util import model_util, data_util, exp_util

"""
Three phases:
1) Feature Removal - Removing features that are causing a domain mismatch.
2) Relabel - identify and relabel training instances to better align with the test set.
3) Data Removal - Remove training instances that cause a domain mismatch.
4) Reweight - upweight and downweight specific training instances to better align with the test set.
"""


def _shap_method(tree, X_test_miss, plot=False):
    """Returns the most impactful features on the specified test instances based on SHAP values."""

    shap_explainer = shap.TreeExplainer(tree)
    test_shap = shap_explainer.shap_values(X_test_miss)
    feature_order = np.argsort(np.sum(np.abs(test_shap), axis=0))[::-1]

    if plot:
        shap.summary_plot(test_shap, X_test_miss)

    return feature_order


def _sexee_method(tree, X_test_miss, X_train, y_train, plot=False, topk=100):
    """
    Returns the most impactful features on the specific test instances by finding the most impactful
    train instances, then getting the SHAP values of those instances.
    """

    # find the most impactful train instances on the test instances
    sexee_explainer = sexee.TreeExplainer(tree, X_train, y_train)
    train_impact = sexee_explainer.get_train_weight()[:topk]
    pos_ndx, pos_val = zip(*train_impact)
    pos_ndx = np.array(pos_ndx)

    shap_explainer = shap.TreeExplainer(tree)
    train_shap = shap_explainer.shap_values(X_train[pos_ndx])
    feature_order = np.argsort(np.sum(np.abs(train_shap), axis=0))[::-1]

    if plot:
        shap.summary_plot(train_shap, X_train[pos_ndx])

    return feature_order


def _retrain(train_order, clf, X_train, y_train, X_test, y_test, k=6):
    """Return model performance as features are removed from the data."""

    auroc = []
    for i in tqdm.tqdm(range(1, k)):
        remove = train_order[:i]
        new_X_train = np.delete(X_train, remove, axis=1)
        new_X_test = np.delete(X_test, remove, axis=1)
        new_tree = clone(clf).fit(new_X_train, y_train)
        yhat = new_tree.predict_proba(new_X_test)[:, 1]
        auroc.append(roc_auc_score(y_test, yhat))

    return auroc


def remove_feature(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100,
                   random_state=69, plot=False, data_dir='data', test_subset=200, n_remove=8):
    """Can we identify features to remove that would improve model performance on the test set."""

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    yhat = tree.predict_proba(X_test)[:, 1]
    og_auroc = roc_auc_score(y_test, yhat)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    # get worst test instance loss
    test_dist = exp_util.instance_loss(tree.predict_proba(X_test), y_test)
    test_dist_ndx = np.argsort(test_dist)[::-1][:test_subset]
    test_dist = test_dist[test_dist_ndx]
    X_test_miss = X_test[test_dist_ndx]

    # plot test instance loss
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(test_dist)), test_dist)
        ax.set_ylabel('l1 loss')
        ax.set_xlabel('test instances sorted by loss')
        plt.show()

    # # get test instance shap values
    feature_order_shap = _shap_method(tree, X_test_miss, plot=plot)
    feature_order_sexee = _sexee_method(tree, X_test_miss, X_train, y_train, plot=plot)

    # get feature order to remove
    num_features = n_remove + 1
    shap_auroc = _retrain(feature_order_shap, clf, X_train, y_train, X_test, y_test, k=num_features)
    sexee_auroc = _retrain(feature_order_sexee, clf, X_train, y_train, X_test, y_test, k=num_features)

    # plot results
    x_labels = [str(x) for x in np.arange(1, num_features)]
    x_labels.insert(0, '0')
    shap_auroc.insert(0, og_auroc)
    sexee_auroc.insert(0, og_auroc)

    fig, ax = plt.subplots()
    ax.axhline(og_auroc, color='k', linestyle='--')
    ax.plot(x_labels, shap_auroc, marker='.', color='orange', label='shap')
    ax.plot(x_labels, sexee_auroc, marker='.', color='green', label='ours')
    ax.set_xlabel('# features removed')
    ax.set_ylabel('auroc')
    ax.legend()
    plt.show()


def relabel(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100,
            random_state=69, timeit=False, iterations=5, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding)

    # compute impact of train instances on test instances
    train_ndx, impact = explainer.train_impact(X_test)
    impact = np.mean(impact, axis=1)

    abs_ndx = np.argsort(np.abs(impact))[::-1][:200]
    abs_train_ndx = train_ndx[abs_ndx]

    print(impact[abs_ndx])

    new_X_train = np.delete(X_train, abs_train_ndx, axis=0)
    new_y_train = np.delete(y_train, abs_train_ndx)

    new_tree = clone(clf).fit(new_X_train, new_y_train)
    model_util.performance(new_tree, new_X_train, new_y_train, X_test, y_test)


def reweight(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100,
             random_state=69, timeit=False, iterations=5, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    new_X_train = X_train.copy()
    new_y_train = y_train.copy()

    sample_weight = np.ones(len(new_y_train))

    missed_indices = np.where(tree.predict(X_test) != y_test)[0]

    for i in range(iterations):

        # get impact from incorrectly predicted test instances
        # a positive impact means the train instances contributed towards the model's prediction
        # a negative impact means the train instances contributed away from the model's prediction
        explainer = sexee.TreeExplainer(tree, new_X_train, new_y_train)
        # missed_indices = np.where(tree.predict(X_test) != y_test)[0]
        impact_incorrect = exp_util.avg_impact(explainer, missed_indices[:1000], X_test, progress=True)
        harmful_train = [impact for impact in impact_incorrect.items() if impact[1] > 0]
        helpful_train = [impact for impact in impact_incorrect.items() if impact[1] < 0]

        mult = 10

        # reweight impactful training samples
        for train_ndx, impact_val in harmful_train:
            sample_weight[train_ndx] -= (impact_val * mult)

        for train_ndx, impact_val in helpful_train:
            sample_weight[train_ndx] -= (impact_val * mult)

        # print(harmful_train, len(harmful_train))
        # print()

        # print(helpful_train, len(helpful_train))
        # print()

        # exit(0)

        # # remove potentially harmful train instances
        # new_X_train = np.delete(new_X_train, harmful_ndx, axis=0)
        # new_y_train = np.delete(new_y_train, harmful_ndx)

        # train new model on cleaned data
        tree = clone(clf).fit(new_X_train, new_y_train, sample_weight=sample_weight)
        model_util.performance(tree, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medifor', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees for ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    parser.add_argument('--iterations', metavar='NUM', type=int, default=5, help='Number of reweighting rounds.')
    args = parser.parse_args()
    print(args)
    # alignment(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.timeit, args.iterations)
    # remove_feature(args.model, args.encoding, args.dataset, args.n_estimators, args.rs)
    relabel(args.model, args.encoding, args.dataset, args.n_estimators, args.rs)
