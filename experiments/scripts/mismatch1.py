"""
Experiment: Replicate domain mismatch experiment from Koh and Liang influence paper.
"""
import argparse

import shap
import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

from util import model_util, data_util, exp_util


def _modify(X_train, y_train, age_0_10_ndx=17):
    """
    Change children age < 10 and readmitted from 3/24 to 3/4.
    https://github.com/kohpangwei/influence-release/blob/master/scripts/hospital_readmission.ipynb
    """

    # Remove from the training set all but one young patients who didn't get readmitted
    remove_ndx = np.where((y_train == -1) & (X_train[:, age_0_10_ndx] == 1))[0][:-1]
    X_train_mod = np.delete(X_train, remove_ndx, axis=0)
    y_train_mod = np.delete(y_train, remove_ndx)

    print('In original data, %s/%s children were readmitted.' % (
          np.sum((y_train == 1) & (X_train[:, age_0_10_ndx] == 1)),
          np.sum((X_train[:, age_0_10_ndx] == 1))))
    print('In modified data, %s/%s children were readmitted.' % (
          np.sum((y_train_mod == 1) & (X_train_mod[:, age_0_10_ndx] == 1)),
          np.sum((X_train_mod[:, age_0_10_ndx] == 1))))

    return X_train_mod, y_train_mod


def mismatch(model='lgb', encoding='leaf_output', dataset='hospital', n_estimators=100,
             random_state=69, plot=False, data_dir='data', age_0_10_ndx=17, topk_train=5,
             verbose=0):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_feature=True)

    # train model on original data
    X_train, X_test, y_train, y_test, label, feature = data
    tree = clone(clf).fit(X_train, y_train)
    if verbose > 0:
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    # introduce domain mismatch and train a model on the modified dataset
    X_train_mod, y_train_mod = _modify(X_train, y_train, age_0_10_ndx=age_0_10_ndx)
    tree_mod = clone(clf).fit(X_train_mod, y_train_mod)
    if verbose > 0:
        model_util.performance(tree_mod, X_train, y_train, X_test, y_test)

    # Find children in the test set and see how predictions change on them
    test_children_ndx = np.where(X_test[:, age_0_10_ndx] == 1)[0]
    y_test_children = y_test[test_children_ndx]
    orig_proba = tree.predict_proba(X_test[test_children_ndx])[:, 1]
    mod_proba = tree_mod.predict_proba(X_test[test_children_ndx])[:, 1]
    results = zip(test_children_ndx, y_test_children, orig_proba, mod_proba)

    if verbose > 0:
        for test_ndx, test_label, orig_pred, mod_pred in results:
            if (orig_pred < 0.5) != (mod_pred < 0.5):
                print('*** ', end='')
            print('index {}, label {}: {:.3f} vs. {:.3f}'.format(test_ndx, test_label, orig_pred, mod_pred))

    # Pick one of those children and find the most influential examples on it
    test_ndx = 1742
    x_test = X_test[test_ndx].reshape(1, -1)
    explainer = sexee.TreeExplainer(tree_mod, X_train_mod, y_train_mod, encoding=encoding, random_state=random_state)
    sv_ndx, sv_impact = explainer.train_impact(x_test)
    sv_ndx, sv_impact = exp_util.sort_impact(sv_ndx, sv_impact)

    sv_ndx = sv_ndx[:topk_train]
    train_children_ndx = np.where(X_train_mod[:, age_0_10_ndx] == 1)[0]
    intersection = np.intersect1d(sv_ndx, train_children_ndx)
    print(np.sort(train_children_ndx))
    print(np.sort(sv_ndx))
    print(np.sort(intersection))

    # visualize impactful train instances using SHAP
    shap_explainer = shap.TreeExplainer(tree_mod)
    test_shap = shap_explainer.shap_values(X_test)
    train_shap = shap_explainer.shap_values(X_train_mod)

    shap.initjs()

    tree_pred = tree_mod.predict(x_test)[0]
    tree_actual = y_test[test_ndx]
    print('Test [{}], predicted: {}, actual: {}'.format(test_ndx, label[tree_pred], label[tree_actual]))

    exp_val = shap_explainer.expected_value if len(label) == 2 else shap_explainer.expected_value[tree_pred]
    shap_val = test_shap[test_ndx] if len(label) == 2 else test_shap[tree_pred][test_ndx]
    display(shap.force_plot(exp_val, shap_val, features=x_test, feature_names=feature))

    print('\nTop {} most impactful train instances:\n'.format(topk_train))
    for train_ndx in sv_ndx:
        x_train = X_train_mod[train_ndx].reshape(1, -1)
        tree_pred = tree_mod.predict(x_train)[0]
        tree_actual = y_train[train_ndx]
        print('Train [{}], predicted: {}, actual: {}'.format(train_ndx, label[tree_pred], label[tree_actual]))

        exp_val = shap_explainer.expected_value if len(label) == 2 else shap_explainer.expected_value[tree_pred]
        shap_val = train_shap[train_ndx] if len(label) == 2 else train_shap[tree_pred][train_ndx]
        display(shap.force_plot(exp_val, shap_val, features=x_train, feature_names=feature))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='hospital', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees for ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    parser.add_argument('--iterations', metavar='NUM', type=int, default=1, help='Number of rounds.')
    parser.add_argument('--topk_train', type=int, default=5, help='Number of train instances to inspect.')
    args = parser.parse_args()
    print(args)
    mismatch(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, topk_train=args.topk_train)
