"""
This script plots the data before and after feature transformations.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def _get_results(args, dataset, tree_kernel='None'):
    """
    Return results.
    """

    res_dir = os.path.join(args.in_dir, dataset, args.tree_type, tree_kernel)
    if not os.path.exists(res_dir):
        return None

    res = {}
    res['train_pos'] = np.load(os.path.join(res_dir, 'train_positive.npy'))
    res['train_neg'] = np.load(os.path.join(res_dir, 'train_negative.npy'))
    res['test_pos'] = np.load(os.path.join(res_dir, 'test_positive.npy'))
    res['test_neg'] = np.load(os.path.join(res_dir, 'test_negative.npy'))
    return res


def _plot_graph(ax, res, xlabel=None, ylabel=None,
                alpha=0.5, s=100, title=''):

    # plot train
    ax.scatter(res['train_neg'][:, 0], res['train_neg'][:, 1],
               color='blue', alpha=alpha, label='train (y=0)',
               marker='+', s=s, rasterized=args.rasterize)

    ax.scatter(res['train_pos'][:, 0], res['train_pos'][:, 1],
               color='red', alpha=alpha, label='train (y=1)',
               marker='1', s=s, rasterized=args.rasterize)

    # plot test
    ax.scatter(res['test_neg'][:, 0], res['test_neg'][:, 1],
               color='cyan', alpha=alpha, label='test (y=0)',
               marker='x', s=s, rasterized=args.rasterize)

    ax.scatter(res['test_pos'][:, 0], res['test_pos'][:, 1],
               color='orange', alpha=alpha, label='test (y=1)',
               marker='2', s=s, rasterized=args.rasterize)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis='both', which='major')
    ax.set_title(title)


def main(args):

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('legend', fontsize=18)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=6)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
    fig, axs = plt.subplots(1, 3, figsize=(width, height))
    axs = axs.flatten()

    d1 = args.dataset[0]
    d1_res_none = _get_results(args, d1)

    if d1_res_none:
        _plot_graph(axs[0], d1_res_none, xlabel='tsne 0',
                    ylabel='tsne 1', title='Original')
        axs[0].legend()

    if args.tree_kernel != 'None':
        d1_res_tree = _get_results(args, d1, tree_kernel=args.tree_kernel)

        if d1_res_tree:
            _plot_graph(axs[1], d1_res_tree, xlabel='tsne 0',
                        ylabel=None, title='Tree kernel')

    if len(args.dataset) > 1:
        d2 = args.dataset[1]
        d2_res_tree = _get_results(args, d2, tree_kernel=args.tree_kernel)

        if d2_res_tree:
            _plot_graph(axs[2], d2_res_tree, xlabel='tsne 0',
                        ylabel=None, title='Different train/test')

    os.makedirs(args.out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.ext)),
                rasterized=args.rasterize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['NC17_EvalPart1', 'nc17_mfc18'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/clustering/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/clustering/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='lgb', help='tree ensemble.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    parser.add_argument('--rasterize', action='store_true', default=False, help='rasterize image.')
    args = parser.parse_args()
    main(args)


class Args:

    dataset = ['NC17_EvalPart1', 'nc17_mfc18']
    in_dir = 'output/clustering/'
    out_dir = 'output/plots/clustering/'

    tree_type = 'lgb'
    tree_kernel = 'leaf_output'

    ext = 'png'
    rasterize = True
