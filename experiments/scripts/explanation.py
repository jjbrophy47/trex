"""
Sample explanation for tree ensembles with SEXEE and SHAP.
"""
import argparse

import shap
import numpy as np

from sexee.explainer import TreeExplainer
from util import model_util, data_util, print_util, exp_util


def example(model='lgb', encoding='leaf_path', dataset='iris', n_estimators=100, random_state=69, timeit=False,
            topk_train=5, topk_test=1, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, return_feature=True, data_dir=data_dir)
    X_train, X_test, y_train, y_test, label, feature = data

    feature = [x[:10] for x in feature]

    # train a tree ensemble
    model = clf.fit(X_train, y_train)
    tree_yhat = model_util.performance(model, X_train, y_train, X_test, y_test)

    # train an svm on learned representations from the tree ensemble
    explainer = TreeExplainer(model, X_train, y_train, encoding=encoding, random_state=random_state, timeit=timeit)
    print(explainer)

    shap_explainer = shap.TreeExplainer(model)
    test_shap = shap_explainer.shap_values(X_test)
    train_shap = shap_explainer.shap_values(X_train)

    # extract predictions
    tree_yhat_train, tree_yhat_test = tree_yhat

    # get worst missed test indices
    test_dist = exp_util.instance_loss(model.predict_proba(X_test), y_test)
    test_dist_ndx = np.argsort(test_dist)[::-1]
    test_dist = test_dist[test_dist_ndx]
    both_missed_test = test_dist_ndx

    # show explanations for missed instances
    for test_ndx in both_missed_test[:topk_test]:
        x_test = X_test[test_ndx]
        train_indices, impact, sim, weight, intercept = explainer.train_impact(x_test, similarity=True, weight=True,
                                                                               intercept=True)
        impact_list = zip(train_indices, impact, sim, weight)
        impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)
        train_indices, impact, sim, weight = zip(*impact_list)
        decision, pred_label = explainer.decision_function(x_test, pred_label=True)
        pred_label = int(pred_label[0])
        print(decision)
        distance = decision[0][pred_label] if decision.ndim == 2 else decision[0]
        print(distance, pred_label)
        print_util.show_test_instance(test_ndx, distance, pred_label, y_test=y_test, label=label)
        print_util.show_train_instances(impact_list, y_train, k=topk_train, label=label, intercept=intercept)

        shap.initjs()

        exp_val = shap_explainer.expected_value if len(label) == 2 else shap_explainer.expected_value[pred_label]
        shap_val = test_shap[test_ndx] if len(label) == 2 else test_shap[pred_label][test_ndx]

        display(shap.force_plot(exp_val, shap_val, features=x_test, feature_names=feature))

        for train_ndx in train_indices[:topk_train]:
            x_train = X_train[train_ndx].reshape(1, -1)
            train_pred = model.predict(x_train)[0]
            print(train_ndx)
            exp_val = shap_explainer.expected_value if len(label) == 2 else shap_explainer.expected_value[train_pred]
            shap_val = train_shap[train_ndx] if len(label) == 2 else train_shap[train_pred][train_ndx]
            display(shap.force_plot(exp_val, shap_val, features=x_train, feature_names=feature))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', metavar='NUM', type=int, default=5, help='Train instances to show.')
    parser.add_argument('--topk_test', metavar='NUM', type=int, default=1, help='Missed test instances to show.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    args = parser.parse_args()
    print(args)
    example(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.timeit,
            args.topk_train, args.topk_test)
