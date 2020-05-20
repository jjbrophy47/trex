"""
Experiment: Remove the training instances that contribute the most
towards the wrongly predicted label for misclassified test instances.
"""
import os
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score

import trex
from utility import model_util
from utility import data_util
from utility import print_util


def experiment(args, logger, out_dir, seed):

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    random_state=seed)
    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir,
                              return_feature=True)
    X_train, X_test, y_train, y_test, label, feature = data

    logger.info('train instances: {:,}'.format(len(X_train)))
    logger.info('test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test, logger=logger)

    if args.scoring == 'auroc':
        original_score = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    else:
        original_score = accuracy_score(y_test, tree.predict(X_test))

    # train TREX
    explainer = trex.TreeExplainer(tree, X_train, y_train,
                                   tree_kernel=args.tree_kernel,
                                   random_state=seed,
                                   kernel_model=args.kernel_model,
                                   kernel_model_kernel=args.kernel_model_kernel)

    # get missed test instances
    missed_indices = np.where(tree.predict(X_test) != y_test)[0]

    np.random.seed(seed)
    explain_indices = np.random.choice(missed_indices, replace=False,
                                       size=int(len(missed_indices) * args.sample_frac))

    logger.info('no. incorrect instances: {:,}'.format(len(missed_indices)))
    logger.info('no. explain instances: {:,}'.format(len(explain_indices)))

    # compute total impact of train instances on test instances
    contributions = explainer.explain(X_test[explain_indices], y=y_test[explain_indices])
    impact_sum = np.sum(contributions, axis=0)

    # get train instances that impact the predictions
    neg_contributors = np.where(impact_sum < 0)[0]
    neg_impact = impact_sum[neg_contributors]
    neg_contributors = neg_contributors[np.argsort(neg_impact)]

    # remove offending train instances in segments and measure performance
    scores, n_removed = [], []

    for i in tqdm.tqdm(range(args.n_iterations + 1)):

        # remove these instances from the train data
        delete_ndx = neg_contributors[:args.n_remove * i]
        new_X_train = np.delete(X_train, delete_ndx, axis=0)
        new_y_train = np.delete(y_train, delete_ndx)

        tree = clone(clf).fit(new_X_train, new_y_train)

        if args.scoring == 'auroc':
            scores.append(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1]))
        else:
            scores.append(accuracy_score(y_test, tree.predict(X_test)))

        n_removed.append(args.n_remove * i)

    result = tree.get_params()
    result[seed] = scores
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # plot results
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(n_removed, scores, fmt='-o', color='green')
    ax.axhline(original_score, linestyle='--', color='k')
    ax.set_xlabel('train instances removed')
    ax.set_ylabel('test {}'.format(args.scoring))
    ax.tick_params(axis='both', which='major')
    plt.show()

    return scores, n_removed, original_score


def main(args):

    # make logger
    dataset = args.dataset

    for i in range(args.repeats):
        seed = args.rs + i
        rs_dir = os.path.join(args.out_dir, dataset, 'rs{}'.format(seed))
        os.makedirs(rs_dir, exist_ok=True)

        logger = print_util.get_logger(os.path.join(rs_dir, '{}.txt'.format(args.dataset)))
        logger.info(args)

        experiment(args, logger, rs_dir, seed=args.rs)

        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/removal/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='lgb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')

    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='kernel model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='similarity kernel')
    parser.add_argument('--true_label', action='store_true', help='train TREX on the true labels.')

    parser.add_argument('--iterations', type=int, default=5, help='Number of rounds.')
    parser.add_argument('--sample_frac', type=float, default=0.1, help='Fraction of test instances to explain.')
    parser.add_argument('--n_remove', type=int, default=50, help='Number of points to remove.')
    parser.add_argument('--scoring', default='auroc', help='Metric to use.')

    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat experiment.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)


class Args:
    dataset = 'mfc19_mfc20'
    data_dir = 'data'
    out_dir = 'output/clustering/'

    tree_type = 'lgb'
    n_estimators = 100
    # max_depth = None
    # C = 0.1

    tree_kernel = 'leaf_output'
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'
    true_label = False

    sample_frac = 0.1
    n_iterations = 5
    n_remove = 50
    scoring = 'auroc'

    repeats = 1
    rs = 1
    verbose = 0
