"""
Sample explanation for tree ensembles with SEXEE.
"""
import argparse
import numpy as np

from sexee.explainer import TreeExplainer
from util import model_util, data_util, print_util, exp_util


def example(model='lgb', encoding='leaf_path', dataset='iris', n_estimators=100, random_state=69, timeit=False,
            topk_train=5, topk_test=1):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state)

    # train a tree ensemble
    model = clf.fit(X_train, y_train)
    tree_yhat = model_util.performance(model, X_train, y_train, X_test, y_test)

    # train an svm on learned representations from the tree ensemble
    explainer = TreeExplainer(model, X_train, y_train, encoding=encoding, random_state=random_state, timeit=timeit)
    print(explainer)

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
        train_ndx, impact, sim, weight, intercept = explainer.train_impact(x_test, similarity=True, weight=True,
                                                                           intercept=True)
        impact_list = zip(train_ndx, impact, sim, weight)
        impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)
        decision, pred_label = explainer.decision_function(x_test, pred_label=True)
        pred_label = int(pred_label[0])
        print_util.show_test_instance(test_ndx, decision[0][pred_label], pred_label, y_test=y_test, label=label)
        print_util.show_train_instances(impact_list, y_train, k=topk_train, label=label, intercept=intercept)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', metavar='NUM', type=int, default=5, help='Train instances to show.')
    parser.add_argument('--topk_test', metavar='NUM', type=int, default=1, help='Missed test instances to show.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    args = parser.parse_args()
    print(args)
    example(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.timeit,
            args.topk_train, args.topk_test)
