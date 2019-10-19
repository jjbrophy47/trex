"""
Experiment: Remove the training instances that contribute the most
towards the wrongly predicted label for misclassified test instances.
"""
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import trex
from utility import model_util, data_util


def _remove_data(model='lgb', dataset='nc17_mfc18', encoding='leaf_output', linear_model='lr', kernel='linear',
                 n_estimators=100, random_state=69, test_size=0.2, iterations=1, n_remove=50, sample_frac=0.1,
                 data_dir='data', verbose=0, scoring='auroc'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    if verbose > 0:
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    if scoring == 'auroc':
        original_score = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    else:
        original_score = accuracy_score(y_test, tree.predict(X_test))

    explainer = trex.TreeExplainer(tree, X_train, y_train, encoding=encoding, kernel=kernel,
                                   random_state=random_state, linear_model=linear_model, dense_output=True)
    if verbose > 0:
        print(explainer)

    # get missed test instances
    missed_indices = np.where(tree.predict(X_test) != y_test)[0]

    np.random.seed(random_state)
    explain_indices = np.random.choice(missed_indices, replace=False, size=int(len(missed_indices) * sample_frac))

    if verbose > 0:
        print(missed_indices, missed_indices.shape)
        print(explain_indices, explain_indices.shape)

    # compute total impact of train instances on test instances
    if verbose > 0:
        print('explaining...')
    contributions = explainer.explain(X_test[explain_indices], y=y_test[explain_indices])
    impact_sum = np.sum(contributions, axis=0)

    # get train instances that impact the predictions
    neg_contributors = np.where(impact_sum < 0)[0]
    neg_impact = impact_sum[neg_contributors]
    neg_contributors = neg_contributors[np.argsort(neg_impact)]

    # remove offending train instances in segments and measure performance
    scores, n_removed = [], []

    for i in range(iterations):

        # remove these instances from the train data
        delete_ndx = neg_contributors[:n_remove * i]
        new_X_train = np.delete(X_train, delete_ndx, axis=0)
        new_y_train = np.delete(y_train, delete_ndx)

        tree = clone(clf).fit(new_X_train, new_y_train)

        if scoring == 'auroc':
            scores.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
        else:
            scores.append(accuracy_score(y_test, tree.predict(X_test)))
        n_removed.append(n_remove * i)

    return scores, n_removed, original_score


def remove_data(model='lgb', dataset='nc17_mfc18', encoding='leaf_output', linear_model='lr', kernel='linear',
                n_estimators=100, random_state=69, test_size=0.2, iterations=1, n_remove=50, sample_frac=0.1,
                data_dir='data', verbose=0, scoring='auroc', repeats=10, out_dir='output/removal'):

    score_list = []
    for i in range(repeats):
        print('run {}'.format(i + 1))
        scores, n_removed, original_score = _remove_data(model=model, encoding=encoding, dataset=dataset,
                                                         n_estimators=n_estimators, random_state=i,
                                                         iterations=iterations, sample_frac=sample_frac,
                                                         n_remove=n_remove, linear_model=linear_model,
                                                         kernel=kernel, scoring=scoring, verbose=verbose,
                                                         data_dir=data_dir)
        score_list.append(scores)

    scores = np.vstack(score_list)
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(n_removed, mean_scores, yerr=std_scores, fmt='-o', color='green')
    ax.axhline(original_score, linestyle='--', color='k')
    ax.set_xlabel('train instances removed', fontsize=24)
    ax.set_ylabel('test {}'.format(scoring), fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'removal.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees for ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--iterations', metavar='NUM', type=int, default=5, help='Number of rounds.')
    parser.add_argument('--sample_frac', type=float, default=0.1, help='Fraction of test instances to explain.')
    parser.add_argument('--n_remove', type=int, default=50, help='Number of points to remove.')
    parser.add_argument('--scoring', default='auroc', help='Metric to use.')
    parser.add_argument('--repeats', type=int, default=10, help='Number of times to repeat experiment.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity.')
    args = parser.parse_args()
    print(args)
    remove_data(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
                random_state=args.rs, iterations=args.iterations, sample_frac=args.sample_frac,
                n_remove=args.n_remove, linear_model=args.linear_model, kernel=args.kernel, scoring=args.scoring,
                repeats=args.repeats, verbose=args.verbose)
