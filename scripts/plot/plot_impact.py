"""
Plots the impact results.
"""
import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util


def main(args):
    print(args)

    # settings
    dataset_list = ['surgical', 'vaccine', 'amazon', 'bank_marketing', 'adult', 'census']

    # label, color, marker, linestyle, zorder
    methods = {}
    methods['klr_og_leaf_path_alpha'] = ['TREX (OG-LP-Alpha)', 'blue', '1', '-', 11]
    methods['klr_og_leaf_path_sim'] = ['TREX (OG-LP-Sim)', 'blue', '1', '--', 10]
    methods['klr_og_leaf_path'] = ['TREX (OG-LP-AlphaSim)', 'blue', '1', ':', 11]
    methods['klr_og_weighted_leaf_path_alpha'] = ['TREX (OG-WLP-Alpha)', 'cyan', '2', '-', 11]
    methods['klr_og_weighted_leaf_path_sim'] = ['TREX (OG-WLP-Sim)', 'cyan', '2', '--', 11]
    methods['klr_og_weighted_leaf_path'] = ['TREX (OG-WLP-AlphaSim)', 'cyan', '2', ':', 11]
    methods['random'] = ['Random', 'red', 'o', '-', 9]
    # methods['maple+'] = ['MAPLE', 'orange', '^', '--', 7]
    methods['maple+_og'] = ['MAPLE (OG)', 'orange', '^', ':', 7]
    methods['leaf_influence'] = ['LeafInfluence', 'black', '.', '-', 1]
    methods['fast_leaf_influence'] = ['FastLeafInfluence', 'black', '.', '--', 1]
    methods['knn_og'] = ['TEKNN (OG)', 'magenta', 'h', '--', 5]
    methods['bacon'] = ['Bacon', 'green', '2', '--', 5]

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['model'] == args.model]
    df = df[df['setting'] == args.setting]
    df = df[df['start_pred'] == args.start_pred]
    df = df[df['n_test'] == args.n_test]

    # plot settings
    util.plot_settings(fontsize=13, markersize=5)

    # inches
    width = 4.8  # Machine Learning journal
    height = util.get_height(width=width, subplots=(2, 3))

    if args.model == 'cb':
        fig, axs = plt.subplots(2, 3, figsize=(width * 1.85, height * 2.95))
    else:
        fig, axs = plt.subplots(2, 3, figsize=(width * 1.95, height * 3.5))

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

                if args.metric == 'proba':
                    ax.set_ylabel(r'Test prob.')

            # add x-axis
            if i == 1:
                ax.set_xlabel('Train data removed (%)')

            # add title
            ax.set_title('Bank Marketing' if dataset == 'bank_marketing' else dataset.capitalize())
            ax.tick_params(axis='both', which='major')

            # plot each method
            for method, (label, color, marker, linestyle, zorder) in methods.items():

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

                # plot only the first 10 checkpoints
                if args.view == 'zoom':
                    metric_mean = metric_mean[:10]
                    metric_sem = metric_mean[:10]
                    removed_pcts = removed_pcts[:10]

                # plot
                yerr = metric_sem if args.view == 'normal' else None
                line = ax.errorbar(removed_pcts, metric_mean, yerr=yerr, marker=marker,
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
    out_dir = os.path.join(args.out_dir,
                           args.model,
                           args.setting,
                           'start_pred_{}'.format(args.start_pred),
                           args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # adjust legend
    if len(lines) <= 4:
        fig.legend(tuple(lines), tuple(labels), loc='center', ncol=6, bbox_to_anchor=(0.5, 0.04))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.225, wspace=0.3)

    else:
        fig.legend(tuple(lines), tuple(labels), loc='center', ncol=3, bbox_to_anchor=(0.5, 0.075))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.265, wspace=0.3)

    # save figure
    fp = os.path.join(out_dir, 'n_test_{}_{}.pdf'.format(args.n_test, args.view))
    plt.savefig(fp)
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/impact/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/impact/', help='output directory.')
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model.')
    parser.add_argument('--metric', type=str, default='proba', help='peformance metric.')
    parser.add_argument('--setting', type=str, default='dynamic', help='evaluation setting.')
    parser.add_argument('--start_pred', type=int, default=1, help='starting prediction.')
    parser.add_argument('--view', type=str, default='normal', help='normal or zoom.')
    parser.add_argument('--n_test', type=int, default=1, help='no. test instances.')

    args = parser.parse_args()
    main(args)
