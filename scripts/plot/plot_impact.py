"""
Plots the ROAR results.
"""
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_height(width, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return height


def main(args):
    print(args)

    # settings
    dataset_list = ['surgical', 'vaccine', 'amazon', 'bank_marketing', 'adult', 'census']
    method_list = ['klr', 'random', 'maple', 'knn', 'leaf_influence', 'fast_leaf_influence', 'maple+']
    color_list = ['blue', 'red', 'orange', 'purple', 'black', 'brown', 'orange']
    label_list = ['TREX', 'Random', 'MAPLE', 'TEKNN', 'LeafInfluence', 'FastLeafInfluence', 'MAPLE+']
    marker_list = ['d', 'o', '^', 'x', '1', '2', '^']
    linestyle_list = ['-', '-', '-', '-', '-', '--', '--']
    zorder_list = [4, 3, 2, 1, 1, 1, 2]

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['model'] == args.model]

    # plot settings
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=13)
    plt.rc('axes', titlesize=13)
    plt.rc('legend', fontsize=13)
    # plt.rc('legend', title_fontsize=11)
    # plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=5)

    # inches
    width = 4.8  # Machine Learning journal
    height = get_height(width=width, subplots=(2, 3))

    if args.model == 'cb':
        fig, axs = plt.subplots(2, 3, figsize=(width * 1.85, height * 2.65))
    else:
        fig, axs = plt.subplots(2, 3, figsize=(width * 1.75, height * 2.35))

    # legend containers
    lines = []
    labels = []

    # dataset incrementer
    k = 0

    # plot datasets
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):

            # extract dataset results
            ax = axs[i][j]
            dataset = dataset_list[k]
            temp_df1 = df[df['dataset'] == dataset]

            #  extract categorical preprocessing results for RF - Amazon
            if args.model == 'rf' and dataset == 'amazon':
                temp_df1 = temp_df1[temp_df1['preprocessing'] == 'categorical']

            # add y-axis
            if j == 0:

                if args.metric == 'proba_diff':
                    ax.set_ylabel(r'|Test prob. $\Delta$|')

            # add x-axis
            if i == 1:
                ax.set_xlabel('Train data removed (%)')

            # add title
            ax.set_title('Bank Marketing' if dataset == 'bank_marketing' else dataset.capitalize())
            ax.tick_params(axis='both', which='major')

            # plot each method
            methods = list(zip(method_list, label_list, color_list, marker_list, linestyle_list, zorder_list))
            for method, label, color, marker, linestyle, zorder in methods:

                # get method results
                temp_df2 = temp_df1[temp_df1['method'] == method]

                if len(temp_df2) == 0:
                    continue

                # extract performance results
                temp_df2 = temp_df2.iloc[0]
                metric_mean = temp_df2['{}_mean'.format(args.metric)]
                metric_sem = temp_df2['{}_sem'.format(args.metric)]
                removed_pcts = temp_df2['remove_pct']

                # convert results from strings to arrays
                metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
                metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
                removed_pcts = np.fromstring(removed_pcts[1: -1], dtype=np.float32, sep=' ')

                # plot
                line = ax.errorbar(removed_pcts, metric_mean, yerr=metric_sem, marker=marker,
                                   linestyle=linestyle, color=color, label=label, zorder=zorder)

                # save for legend
                if i == 0 and j == 0:
                    lines.append(line)
                    labels.append(label)

            # increment dataset
            k += 1

            # set x-aixs limits
            ax.set_xlim(left=0, right=None)

    # create output directory
    out_dir = os.path.join(args.out_dir, args.model, args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # adjust legend
    if len(lines) <= 4:
        fig.legend(tuple(lines), tuple(labels), loc='center', ncol=6, bbox_to_anchor=(0.5, 0.04))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.225, wspace=0.3)

    else:
        fig.legend(tuple(lines), tuple(labels), loc='center', ncol=3, bbox_to_anchor=(0.5, 0.06))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.25, wspace=0.3)

    # save figure
    fp = os.path.join(out_dir, 'all_datasets.pdf')
    plt.savefig(fp)
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/impact/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/impact/', help='output directory.')
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model.')
    parser.add_argument('--metric', type=str, default='proba_diff', help='peformance metric.')

    args = parser.parse_args()
    main(args)
