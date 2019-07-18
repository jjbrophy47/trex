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
from sklearn.preprocessing import minmax_scale

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


def _identify_instances(impact_sum, negative=False, threshold=0):
    """Idenitfies the support vectors that have a net positive / negative impact."""

    # get train instances that positively / negatively impact the predictions
    if negative:
        target_ndx = np.where(impact_sum < threshold)[0]
    else:
        target_ndx = np.where(impact_sum > threshold)[0]

    return target_ndx


def relabel(model='lgb', encoding='leaf_path', dataset='nc17_mfc18', n_estimators=100,
            random_state=69, test_size=0.2, iterations=1, n_points=50, data_dir='data',
            negative=False, threshold=0):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir,
                                                                 test_size=test_size)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    original_auroc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

    new_X_train = X_train.copy()
    new_y_train = y_train.copy()

    if iterations > 1:
        auroc, n_iterations = [], []

        for i in tqdm.tqdm(range(iterations)):

            explainer = sexee.TreeExplainer(tree, new_X_train, new_y_train, encoding=encoding,
                                            random_state=random_state)

            # compute total impact of train instances on test instances
            sv_ndx, sv_impact, weight = explainer.train_impact(X_test, weight=True)
            impact_sum = np.sum(sv_impact, axis=1)

            # get impactful train instances
            target_ndx = _identify_instances(impact_sum, negative=negative, threshold=threshold)
            flip_ndx = sv_ndx[target_ndx]

            # flip the label of these train instances
            new_y_train = data_util.flip_labels_with_indices(new_y_train, flip_ndx)

            tree = clone(clf).fit(new_X_train, new_y_train)
            auroc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
            n_iterations.append(i + 1)

        n_iterations.insert(0, 0)
        auroc.insert(0, original_auroc)

        fig, ax = plt.subplots()
        ax.plot(n_iterations, auroc, color='green', marker='.')
        ax.axhline(original_auroc, linestyle='--', color='k')
        ax.set_xlabel('iteration')
        ax.set_ylabel('test auroc')
        plt.show()

    else:

        explainer = sexee.TreeExplainer(tree, new_X_train, new_y_train, encoding=encoding,
                                        random_state=random_state)

        # compute total impact of train instances on test instances
        sv_ndx, sv_impact, weight = explainer.train_impact(X_test, weight=True)
        impact_sum = np.sum(sv_impact, axis=1)

        # get impactful train instances
        target_ndx = _identify_instances(impact_sum, negative=negative, threshold=threshold)
        flip_ndx = sv_ndx[target_ndx]
        target_impact = impact_sum[target_ndx]
        remove_ndx, pos_impact = exp_util.sort_impact(target_ndx, target_impact, ascending=False)

        # remove offending train instances in segments and measure performance
        step_size = int(len(flip_ndx) / n_points)
        auroc, n_flipped = [], []

        for i in tqdm.tqdm(range(step_size, len(flip_ndx) + step_size, step_size)):

            # remove these instances from the train data
            new_y_train = data_util.flip_labels_with_indices(y_train, flip_ndx[:i])

            tree = clone(clf).fit(new_X_train, new_y_train)
            auroc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
            n_flipped.append(i)

        n_flipped.insert(0, 0)
        auroc.insert(0, original_auroc)

        fig, ax = plt.subplots()
        ax.plot(n_flipped, auroc, color='green', marker='.')
        ax.axhline(original_auroc, linestyle='--', color='k')
        ax.set_xlabel('train instances flipped')
        ax.set_ylabel('test auroc')
        plt.show()


def remove_data(model='lgb', encoding='leaf_path', dataset='nc17_mfc18', n_estimators=100,
                random_state=69, test_size=0.2, iterations=1, n_points=150, data_dir='data',
                verbose=0, negative=False, threshold=0, absolute=False):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir,
                                                                 test_size=test_size)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)
    original_auroc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

    new_X_train = X_train.copy()
    new_y_train = y_train.copy()

    if iterations > 1:
        auroc, n_iter = [], []

        for i in tqdm.tqdm(range(iterations)):
            explainer = sexee.TreeExplainer(tree, new_X_train, new_y_train, encoding=encoding,
                                            random_state=random_state)
            if verbose > 0:
                print(explainer)

            # compute total impact of train instances on test instances
            sv_ndx, sv_impact, weight = explainer.train_impact(X_test, weight=True)
            impact_sum = np.sum(sv_impact, axis=1)

            # get train instances that impact the predictions
            target_ndx = _identify_instances(impact_sum, negative=negative, threshold=threshold)
            remove_ndx = sv_ndx[target_ndx]

            # remove these instances from the train data
            new_X_train = np.delete(new_X_train, remove_ndx, axis=0)
            new_y_train = np.delete(new_y_train, remove_ndx)

            tree = clone(clf).fit(new_X_train, new_y_train)
            auroc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
            n_iter.append(i + 1)

        n_iter.insert(0, 0)
        auroc.insert(0, original_auroc)

        fig, ax = plt.subplots()
        ax.plot(n_iter, auroc, color='green', marker='.')
        ax.axhline(original_auroc, linestyle='--', color='k')
        ax.set_xlabel('iteration')
        ax.set_ylabel('test auroc')
        plt.show()

    else:
        explainer = sexee.TreeExplainer(tree, new_X_train, new_y_train, encoding=encoding,
                                        random_state=random_state)
        if verbose > 0:
            print(explainer)

        # compute total impact of train instances on test instances
        sv_ndx, sv_impact, weight = explainer.train_impact(X_test, weight=True)
        impact_sum = np.sum(sv_impact, axis=1)

        # get train instances that impact the predictions
        if absolute:
            remove_ndx, pos_impact = exp_util.sort_impact(sv_ndx, sv_impact, ascending=False)
            print(remove_ndx)
            print(pos_impact)
        else:
            target_ndx = _identify_instances(impact_sum, negative=negative, threshold=threshold)
            remove_ndx = sv_ndx[target_ndx]
            target_impact = impact_sum[target_ndx]
            remove_ndx, pos_impact = exp_util.sort_impact(remove_ndx, target_impact, ascending=False)

        # remove offending train instances in segments and measure performance
        step_size = int(len(remove_ndx) / n_points)
        auroc, n_removed = [], []

        for i in tqdm.tqdm(range(step_size, len(remove_ndx) + step_size, step_size)):

            # remove these instances from the train data
            delete_ndx = remove_ndx[:i]
            new_X_train = np.delete(X_train, delete_ndx, axis=0)
            new_y_train = np.delete(y_train, delete_ndx)

            tree = clone(clf).fit(new_X_train, new_y_train)
            auroc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
            n_removed.append(i)

        n_removed.insert(0, 0)
        auroc.insert(0, original_auroc)

        fig, ax = plt.subplots()
        ax.plot(n_removed, auroc, color='green', marker='.')
        ax.axhline(original_auroc, linestyle='--', color='k')
        ax.set_xlabel('train instances removed')
        ax.set_ylabel('test auroc')
        plt.show()


def _scale(sample_weight, impact_sum, sv_ndx, together=False, scale_max=0.5,
           negative=False, threshold=0):

    # reweight support vectors, positive impacts get lower weight, negative impacts get higher weight
    if together:
        impact_sum_scaled = minmax_scale(impact_sum, feature_range=(-scale_max, scale_max))
        sample_weight[sv_ndx] -= impact_sum_scaled

    else:

        # downweight positively impactful train instances if negative is False
        pos_sv_ndx = _identify_instances(impact_sum, negative=False, threshold=threshold)
        pos_ndx = sv_ndx[pos_sv_ndx]
        pos_impact = minmax_scale(impact_sum[pos_sv_ndx], feature_range=(0, scale_max))

        # upweight negatively impactful train instances if negative is False
        neg_sv_ndx = _identify_instances(impact_sum, negative=True, threshold=threshold)
        neg_ndx = sv_ndx[neg_sv_ndx]
        neg_impact = minmax_scale(impact_sum[neg_sv_ndx], feature_range=(-scale_max, 0))

        if negative:
            sample_weight[pos_ndx] += pos_impact
            sample_weight[neg_ndx] += neg_impact
        else:
            sample_weight[pos_ndx] -= pos_impact
            sample_weight[neg_ndx] -= neg_impact

    return sample_weight


def reweight(model='lgb', encoding='leaf_output', dataset='nc17_mfc18', n_estimators=100,
             random_state=69, test_size=0.2, iterations=1, data_dir='data', scale_max=0.5,
             scale_together=False, negative=False):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir,
                                                                 test_size=test_size)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    original_auroc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])

    new_X_train = X_train.copy()
    new_y_train = y_train.copy()
    sample_weight = np.ones(len(new_y_train))

    auroc, n_iter = [], []
    for i in tqdm.tqdm(range(iterations)):

        # get impact of train instances on test instances
        explainer = sexee.TreeExplainer(tree, new_X_train, new_y_train, encoding=encoding,
                                        random_state=random_state)

        # compute total impact of train instances on test instances
        sv_ndx, sv_impact, weight = explainer.train_impact(X_test, weight=True)
        impact_sum = np.sum(sv_impact, axis=1)

        sample_weight = _scale(sample_weight, impact_sum, sv_ndx, together=scale_together,
                               scale_max=scale_max, negative=negative)

        # train new model on reweighted data
        tree = clone(clf).fit(new_X_train, new_y_train, sample_weight=sample_weight)
        auroc.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
        n_iter.append(i + 1)

    n_iter.insert(0, 0)
    auroc.insert(0, original_auroc)

    fig, ax = plt.subplots()
    ax.plot(n_iter, auroc, color='green', marker='.')
    ax.axhline(original_auroc, linestyle='--', color='k')
    ax.set_xlabel('iteration')
    ax.set_ylabel('test auroc')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees for ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    parser.add_argument('--iterations', metavar='NUM', type=int, default=1, help='Number of rounds.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of the test set.')
    parser.add_argument('--negative', action='store_true', help='Remove / Relabel negative impact instances.')
    parser.add_argument('--absolute', action='store_true', help='Sort train instances by absolute value.')
    parser.add_argument('--experiment', default='relabel', help='feature_remove, data_remove, relabel, or reweight.')
    parser.add_argument('--n_points', default=50, help='Number of points to plot.')
    args = parser.parse_args()
    print(args)

    if args.experiment == 'remove_feature':
        remove_feature(args.model, args.encoding, args.dataset, args.n_estimators, args.rs)
    elif args.experiment == 'relabel':
        relabel(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.test_size,
                args.iterations, negative=args.negative)
    elif args.experiment == 'remove_data':
        remove_data(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.test_size,
                    args.iterations, negative=args.negative, absolute=args.absolute, n_points=args.n_points)
    elif args.experiment == 'reweight':
        reweight(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.test_size,
                 args.iterations, negative=args.negative)
    else:
        exit('experiment {} unrecognized'.format(args.experiment))
