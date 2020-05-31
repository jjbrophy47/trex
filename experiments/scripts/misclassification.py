"""
Explanation of missclassified test instances for the NC17_EvalPart1 (train) and
MFC18_EvalPart1 (test) dataset using TREX. Visualizes the most important feature
from the raw data perspective (positive vs negative), then weighting it using the weights
for a global explanation then weighting it using similarity x abs(weight) for a local explanation.
This also plots the weight distribution, as well as thr similarity vs weight distribution
for a single test instance.
"""
import os
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner

import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import trex
from utility import model_util
from utility import data_util
from utility import print_util
from utility import exp_util


def _get_top_features(x, shap_vals, feature_list, k=1):
    """
    Parameters
    ----------
    x: 1d array like
        Feature values for this instance.
    shap_vals: 1d array like
        Feature contributions to the prediction.
    feature: 1d array like
        Feature names.
    k: int (default=5)
        Only keep the top k features.

    Returns a list of (feature_name, feature_value, feature_shap) tuples.
    """
    assert len(x) == len(shap_vals) == len(feature_list)
    shap_sort_ndx = np.argsort(np.abs(shap_vals))[::-1]
    return list(zip(feature_list[shap_sort_ndx], x[shap_sort_ndx],
                    shap_vals[shap_sort_ndx]))[:k]


def _plot_feature_histograms(args, results, out_dir):
    """
    Plot the density of the most important feature weighted
    by training instance importance and similarity to the test instance.
    """

    train_feature_vals = results['train_feature_vals']
    train_feature_bins = results['train_feature_bins']
    train_pos_ndx = results['train_pos_ndx']
    train_neg_ndx = results['train_neg_ndx']
    train_weight = results['train_weight']
    train_sim = results['train_sim']

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # unweighted
    ax = axs[0]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha, label='positive instances')
    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha, label='negative instances')
    ax.set_xlabel('value')
    ax.set_ylabel('density')
    ax.set_title('Unweighted')
    ax.set_xlim(-0.25, 1.25)
    ax.legend()
    ax.tick_params(axis='both', which='major')

    # weighted by TREX's global weights
    ax = axs[1]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha,
            weights=np.abs(train_weight)[train_pos_ndx])

    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha,
            weights=np.abs(train_weight)[train_neg_ndx])

    ax.set_xlabel('value')
    ax.set_title(r'|$\alpha$|',)
    ax.set_xlim(-0.25, 1.25)
    ax.tick_params(axis='both', which='major')

    # weighted by TREX's global weights * similarity to the test instance
    train_sim_weight = train_weight * train_sim
    ax = axs[2]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha,
            weights=np.abs(train_sim_weight)[train_pos_ndx])

    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha,
            weights=np.abs(train_sim_weight)[train_neg_ndx])
    ax.set_xlabel('value')
    ax.set_title(r'|$\alpha$| * Similarity')
    ax.set_xlim(-0.25, 1.25)
    ax.tick_params(axis='both', which='major')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_distribution.{}'.format(args.ext)))


def _plot_instance_histograms(args, results, out_dir):
    """
    Plot TREX's:
      weight distribution and
      similarity * weight distribution.
    """
    train_weight = results['train_weight']
    train_sim = results['train_sim']

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # plot weight distribution for the training samples
    ax = axs[0]
    sns.distplot(train_weight, color='orange', ax=axs[0], kde=args.kde)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('density')
    ax.set_title('(a)')
    ax.tick_params(axis='both', which='major')
    if args.xlim:
        ax.set_xlim(-args.xlim, args.xlim)

    # plot similarity x weight for the training samples
    ax = axs[1]
    sns.distplot(train_sim * train_weight, color='green', ax=ax, kde=args.kde)
    ax.set_xlabel(r'$\alpha$ * similarity')
    ax.set_title('(b)')
    ax.tick_params(axis='both', which='major')
    if args.xlim:
        ax.set_xlim(-args.xlim, args.xlim)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weight_distribution.{}'.format(args.ext)))


def experiment(args, logger, out_dir, seed):

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=seed)

    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir,
                              return_feature=True)
    X_train, X_test, y_train, y_test, label, feature = data

    logger.info('train instances: {:,}'.format(len(X_train)))
    logger.info('test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test, logger=logger)

    # train TREX
    logger.info('building TREX...')
    explainer = trex.TreeExplainer(tree, X_train, y_train,
                                   tree_kernel=args.tree_kernel,
                                   random_state=seed,
                                   kernel_model=args.kernel_model,
                                   kernel_model_kernel=args.kernel_model_kernel,
                                   true_label=args.true_label)

    # extract predictions
    logger.info('generating predictions...')
    y_test_pred_tree = tree.predict(X_test)
    y_test_pred_trex = explainer.predict(X_test)

    logger.info('generating probabilities...')
    y_test_proba_tree = tree.predict_proba(X_test)[:, 1]
    if args.kernel_model == 'lr':
        y_test_proba_trex = explainer.predict_proba(X_test)[:, 1]

    # get worst missed test index
    test_dist = exp_util.instance_loss(tree.predict_proba(X_test), y_test)
    test_dist_ndx = np.argsort(test_dist)[::-1]

    # geta random incorrectly predicted test instance
    np.random.seed(seed)
    rand_num = np.random.choice(1000)
    test_ndx = test_dist_ndx[rand_num]
    x_test = X_test[[test_ndx]]

    # show explanations for missed instances
    logger.info('\ntest index: {}, label: {}'.format(test_ndx, y_test[test_ndx]))
    logger.info('tree pred: {} ({:.3f})'.format(y_test_pred_tree[test_ndx],
                                                y_test_proba_tree[test_ndx]))
    if args.kernel_model == 'lr':
        logger.info('TREX pred: {} ({:.3f})'.format(y_test_pred_trex[test_ndx],
                                                    y_test_proba_trex[test_ndx]))
    else:
        logger.info('TREX pred: {}'.format(y_test_pred_trex[test_ndx]))

    # obtain most important features
    logger.info('\ncomputing most influential features...')
    shap_explainer = shap.TreeExplainer(tree)
    test_shap = shap_explainer.shap_values(x_test)
    top_features = _get_top_features(x_test[0], test_shap[0], feature)

    # get positive and negative training samples
    train_pos_ndx = np.where(y_train == 1)[0]
    train_neg_ndx = np.where(y_train == 0)[0]

    # get global weights and similarity to the test instance
    train_weight = explainer.get_weight()[0]
    sim = explainer.similarity(X_test[[test_ndx]])[0]

    logger.info('collecting results...')
    results = explainer.get_params()
    results['test_ndx'] = test_ndx
    results['train_pos_ndx'] = train_pos_ndx
    results['train_neg_ndx'] = train_neg_ndx
    results['train_weight'] = train_weight
    results['train_sim'] = sim

    # explain features
    for target_feature, val, shap_val in top_features:
        feat_ndx = np.where(target_feature == feature)[0][0]
        logger.info('{}, index: {}, val: {}, shap val: {}'.format(
                    target_feature, feat_ndx, val, shap_val))

        train_feature_bins = np.histogram(X_train[:, feat_ndx], bins=args.max_bins)[1]

        results['target_feature'] = target_feature
        results['train_feature_vals'] = X_train[:, feat_ndx]
        results['train_feature_bins'] = train_feature_bins

        _plot_feature_histograms(args, results, out_dir)

    _plot_instance_histograms(args, results, out_dir)
    np.save(os.path.join(out_dir, 'results.npy'), results)


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, args.tree_type,
                           args.tree_kernel, 'rs{}'.format(args.rs))
    os.makedirs(out_dir, exist_ok=True)
    print(out_dir)

    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/misclassification/', help='output directory.')

    parser.add_argument('--max_bins', type=int, default=40, help='number of bins for feature values.')
    parser.add_argument('--alpha', type=float, default=0.5, help='transparency value.')
    parser.add_argument('--xlim', type=float, default=None, help='x limits on instance plots.')
    parser.add_argument('--kde', action='store_true', default=None, help='plot kde on weight distribution.')

    parser.add_argument('--tree_type', type=str, default='lgb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth.')

    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='kernel model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='similarity kernel')
    parser.add_argument('--true_label', action='store_true', default=False, help='train TREX on the true labels.')

    parser.add_argument('--ext', type=str, default='png', help='output image format.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)


class Args:
    dataset = 'nc17_mfc18'
    data_dir = 'data'
    out_dir = 'output/misclassification/'

    max_bins = 40
    alpha = 0.5
    xlim = 0.05
    kde = False

    tree_type = 'lgb'
    n_estimators = 100
    max_depth = None

    tree_kernel = 'leaf_output'
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'
    true_label = False

    ext = 'pdf'

    rs = 1
    verbose = 0
