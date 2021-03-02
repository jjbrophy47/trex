"""
Plots the ROAR results.
"""
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def main(args):
    print(args)

    method_list = ['random', 'klr-leaf_output', 'maple', 'knn-leaf_output']
    color_list = ['red', 'blue', 'orange', 'purple']
    label_list = ['Random', 'TREX-KLR', 'MAPLE', 'TEKNN']
    marker_list = ['o', 'd', '^', 'x']

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['dataset'] == args.dataset]
    df = df[df['model'] == args.model]

    # result containers
    lines = []

    # plot each method
    fig, ax = plt.subplots()
    methods = list(zip(method_list, label_list, color_list, marker_list))

    for method, label, color, marker in methods:

        # get method results
        temp_df = df[df['method'] == method]

        if len(temp_df) == 0:
            continue

        # extract performance results
        temp_df = temp_df.iloc[0]
        metric_mean = temp_df['{}s_mean'.format(args.metric)]
        metric_sem = temp_df['{}s_sem'.format(args.metric)]
        removed_pcts = temp_df['removed_pcts']

        # convert results from strings to arrays
        metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
        metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
        removed_pcts = np.fromstring(removed_pcts[1: -1], dtype=np.float32, sep=' ')

        # plot
        line = ax.errorbar(removed_pcts, metric_mean, yerr=metric_sem,
                           marker=marker, color=color, label=label)
        lines.append(line)

        # add metric-specific info
        if args.metric in ['acc', 'auc']:
            label = 'Acc.' if args.metric == 'acc' else 'AUC'
            ax.set_ylabel('Test {}'.format(label))

        elif 'proba_delta' in args.metric:
            ax.set_ylabel(r'Avg. test prob. $\Delta$')

        else:
            raise ValueError('unknown metric: {}'.format(args.metric))

    # adjust plot
    ax.set_xlabel('Train data removed (%)')
    ax.legend()
    plt.tight_layout()

    # create output directory
    out_dir = os.path.join(args.out_dir, args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # adjust and save plot
    fp = os.path.join(out_dir, '{}.pdf'.format(args.dataset))
    plt.savefig(fp, bbox_inches='tight')

    print('saving to {}...'.format(fp))

    exit(0)






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
        fig, axs = plt.subplots(2, 2, figsize=(width, height * 1.15), sharex='col')

    else:
        plt.rc('xtick', labelsize=17)
        plt.rc('ytick', labelsize=17)
        plt.rc('axes', labelsize=22)
        plt.rc('axes', titlesize=22)
        plt.rc('legend', fontsize=20)
        plt.rc('legend', title_fontsize=11)
        plt.rc('lines', linewidth=2)
        plt.rc('lines', markersize=8)

        width = 5.5  # Neurips 2020
        width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
        fig, axs = plt.subplots(1, max(2, len(args.dataset)),
                                figsize=(width, height * 1.1), sharex=True)
    axs = axs.flatten()

    n_pts = 10

    ylabel = 'Test {}'.format(metric_mapping[args.metric])
    xlabel = '% train data removed'

    lines = []
    lines_ndx = []
    for i, dataset in enumerate(args.dataset):
        ax = axs[i]

        for j, method in enumerate(method_list):
            res = get_results(args, dataset, method)

            if res is not None:
                res_mean, res_sem, pcts = res
                line = ax.errorbar(pcts[:n_pts], res_mean[:n_pts], yerr=res_sem[:n_pts],
                                   marker=markers[j], color=colors[j], label=labels[j])

                if i == 0:
                    lines.append(line[0])
                    lines_ndx.append(j)

        if args.two_col:
            if i % 2 == 0:
                axs[i].set_ylabel(ylabel)
            if i > 1:
                axs[i].set_xlabel(xlabel)

        else:

            if i == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

        if i == 1:
            ax.legend(frameon=False)

        ax.set_title(dataset.capitalize())
        ax.tick_params(axis='both', which='major')

    out_dir = os.path.join(args.out_dir, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()

    if not args.two_col:
        fig.subplots_adjust(wspace=0.25, hspace=0.05)

    plt.savefig(os.path.join(out_dir, 'roar.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/roar/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='tree type.')
    parser.add_argument('--metric', type=str, default='acc', help='acc, auc, avg_proba_delta, median_proba_delta')

    args = parser.parse_args()
    main(args)
