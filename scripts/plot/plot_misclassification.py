"""
Plots the misclassification results.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def get_results(args):
    in_dir = os.path.join(args.in_dir, args.dataset, args.tree_type,
                          args.tree_kernel, 'rs{}'.format(args.rs))
    results_path = os.path.join(in_dir, 'results.npy')
    assert os.path.exists(results_path)
    result = np.load(results_path, allow_pickle=True)[()]
    return result


def main(args):

    # matplotlib settings
    plt.rc('font', family='serif')

    if args.two_col:
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)
        plt.rc('axes', labelsize=11)
        plt.rc('axes', titlesize=11)
        plt.rc('legend', fontsize=11)
        plt.rc('legend', title_fontsize=11)
        plt.rc('lines', linewidth=1)
        plt.rc('lines', markersize=3)

        width = 3.25  # Two column style
        width, height = set_size(width=width * 2, fraction=1, subplots=(2, 2))
        fig, axs = plt.subplots(2, 2, figsize=(width, height))

    else:
        # matplotlib settings
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.rc('axes', labelsize=21)
        plt.rc('axes', titlesize=21)
        plt.rc('legend', fontsize=19)
        plt.rc('legend', title_fontsize=11)
        plt.rc('lines', linewidth=1)
        plt.rc('lines', markersize=6)

        width = 5.5  # Neurips 2020
        width, height = set_size(width=width * 3, fraction=1, subplots=(1, 4))
        fig, axs = plt.subplots(1, 4, figsize=(width, height * 1.25))

    axs = axs.flatten()

    results = get_results(args)

    train_feature_vals = results['train_feature_vals']
    train_feature_bins = results['train_feature_bins']
    train_pos_ndx = results['train_pos_ndx']
    train_neg_ndx = results['train_neg_ndx']
    train_weight = results['train_weight']
    train_sim = results['train_sim']
    feature_name = results['target_feature']
    test_val = results['test_val']

    train_sim_weight = train_weight * train_sim

    # gamma vs alpha
    print('plotting gamma vs alpha...')
    ax = axs[0]
    xy = np.vstack([train_weight, train_sim])
    z = gaussian_kde(xy)(xy)
    ax.scatter(train_weight, train_sim, c=z, s=20, edgecolor='', rasterized=args.rasterize)
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.set_ylabel(r'$\gamma$')
    ax.set_xlabel(r'$\alpha$')

    # unweighted
    print('plotting unweighted...')
    ax = axs[1]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha, label='positive instances')
    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha, label='negative instances')
    ax.axvline(test_val, color='k', linestyle='--')
    ax.set_xlabel(feature_name.capitalize())
    ax.set_ylabel('Density')
    ax.set_title('Unweighted')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major')

    # weighted by TREX's global weights
    print('plotting weighted by global weights...')
    ax = axs[2]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha,
            weights=train_weight[train_pos_ndx])

    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha,
            weights=train_weight[train_neg_ndx])
    ax.axvline(test_val, color='k', linestyle='--')
    ax.set_ylabel('Density')
    ax.set_xlabel(feature_name.capitalize())
    ax.set_title(r'Weighted by $\alpha$',)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major')

    # weighted by TREX's global weights * similarity to the test instance
    print('plotting weighted by weight * similarity...')
    train_sim_weight = train_weight * train_sim
    ax = axs[3]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha,
            weights=train_sim_weight[train_pos_ndx],
            label='pos samples')

    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha,
            weights=train_sim_weight[train_neg_ndx],
            label='neg samples')
    ax.axvline(test_val, color='k', linestyle='--')
    ax.legend()
    ax.set_ylabel('Density')
    ax.set_xlabel(feature_name.capitalize())
    ax.set_title(r'Weighted by $\alpha * \gamma$')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major')

    # save plot
    out_dir = os.path.join(args.out_dir, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()

    if not args.two_col:
        fig.subplots_adjust(wspace=0.25, hspace=0.05)

    plt.savefig(os.path.join(out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/misclassification/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/misclassification/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree type.')
    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='tree kernel.')

    # plot settings
    parser.add_argument('--alpha', type=float, default=0.5, help='transparency value.')
    parser.add_argument('--rasterize', action='store_true', default=False, help='rasterize dense instance plots.')

    parser.add_argument('--two_col', action='store_true', default=False, help='format into two columns.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)


class Args:

    dataset = ['churn', 'amazon', 'adult', 'census']
    in_dir = 'output/roar/'
    out_dir = 'output/plots/roar/'

    tree_type = 'cb'
    tree_kernel = 'tree_output'

    alpha = 0.5
    rasterize = False

    metric = 'acc'
    rs = [1]
    ext = 'png'
    verbose = 0
