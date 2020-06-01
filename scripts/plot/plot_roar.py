"""
Plots the ROAR results.
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


def get_results(args, dataset, method, score_ndx=0):

    # get results
    in_dir = os.path.join(args.in_dir, dataset, args.tree_type, args.tree_kernel)
    pct_path = os.path.join(in_dir, 'percentages.npy')
    method_path = os.path.join(in_dir, '{}.npy'.format(method))

    if not os.path.exists(method_path):
        return None

    pcts = np.load(pct_path)
    res = np.load(method_path)[score_ndx]

    return res, pcts


def main(args):

    method_list = ['trex_lr', 'random', 'maple', 'leafinfluence', 'teknn']
    colors = ['blue', 'red', 'orange', 'black', 'purple']
    labels = ['TREX', 'Random', 'MAPLE', 'LeafInfluence', 'TE-KNN']
    markers = ['1', '*', 'd', 'h', 'o']
    metric_mapping = {'auc': 'AUROC', 'acc': 'Accuracy'}
    metric_ndx_mapping = {'auc': 0, 'acc': 1}

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('legend', fontsize=20)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=8)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
    fig, axs = plt.subplots(1, max(2, len(args.dataset)),
                            figsize=(width, height), sharex=True)
    axs = axs.flatten()

    lines = []
    lines_ndx = []
    for i, dataset in enumerate(args.dataset):
        ax = axs[i]

        for j, method in enumerate(method_list):
            res = get_results(args, dataset, method,
                              score_ndx=metric_ndx_mapping[args.metric])

            if res is not None:
                res, pcts = res
                line = ax.errorbar(pcts[:len(res)], res,
                                   marker=markers[j], color=colors[j], label=labels[j])

                if i == 0:
                    lines.append(line[0])
                    lines_ndx.append(j)

        if i == 0:
            ax.set_ylabel('Test {}'.format(metric_mapping[args.metric]))
        if i == 1:
            ax.legend()
        ax.set_title(dataset.capitalize())
        ax.set_xlabel('% train data removed')
        ax.tick_params(axis='both', which='major')

    os.makedirs(args.out_dir, exist_ok=True)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.05)
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/roar/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree type.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='kernel model.')

    parser.add_argument('--metric', type=str, default='acc', help='predictive metric.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1], help='dataset to explain.')
    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    args = parser.parse_args()
    main(args)


class Args:

    dataset = ['churn', 'amazon', 'adult', 'census']
    in_dir = 'output/roar/'
    out_dir = 'output/plots/roar/'

    tree_type = 'cb'
    tree_kernel = 'leaf_output'
    kernel_model = 'lr'

    rs = [1]
    ext = 'png'
