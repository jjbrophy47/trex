"""
Plots the ROAR results.
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

    methods = {}
    methods['klr'] = ['TREX', 'blue', '1', '-', 11]  # label, color, marker, linestyle, zorder
    methods['random'] = ['Random', 'red', 'o', '-', 9]
    methods['random_minority'] = ['Random (minority class)', 'cyan', 'o', '--', 9]
    methods['random_majority'] = ['Random (majority label)', 'magenta', 'o', '--', 9]
    methods['random_pred'] = ['Random (pred. label)', 'green', 'o', '--', 9]
    methods['maple'] = ['MAPLE', 'orange', '>', '-', 7]
    methods['maple+'] = ['MAPLE+', 'orange', '^', '--', 7]
    methods['fast_leaf_influence'] = ['FastLeafInfluence', 'black', '.', '--', 1]
    methods['knn'] = ['TEKNN', 'yellow', 'h', '-', 5]

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['model'] == args.model]

    # plot settings
    util.plot_settings(fontsize=13, markersize=5)

    # inches
    width = 4.8  # Machine Learning journal
    height = util.get_height(width=width, subplots=(2, 3))
    fig, axs = plt.subplots(2, 3, figsize=(width * 1.75, height * 2.95))

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
                if args.metric in ['acc', 'auc']:
                    label = 'accuracy' if args.metric == 'acc' else 'AUC'
                    ax.set_ylabel('Test {}'.format(label))

                elif args.metric == 'avg_proba_delta':
                    ax.set_ylabel(r'Avg. |test prob. $\Delta$|')

                elif args.metric == 'median_proba_delta':
                    ax.set_ylabel(r'Median |test prob. $\Delta$|')

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
                metric_mean = temp_df2['{}s_mean'.format(args.metric)]
                metric_sem = temp_df2['{}s_sem'.format(args.metric)]
                removed_pcts = temp_df2['remove_pcts']

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

            # set x-axis limits
            ax.set_xlim(left=0, right=None)

    # create output directory
    out_dir = os.path.join(args.out_dir, args.model, args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # adjust legend
    fig.legend(tuple(lines), tuple(labels), loc='center', ncol=3, bbox_to_anchor=(0.5, 0.075))

    # adjust figure
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, wspace=0.3)

    # save figure
    fp = os.path.join(out_dir, 'all_datasets.pdf')
    plt.savefig(fp)
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/roar/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model.')
    parser.add_argument('--metric', type=str, default='avg_proba_delta',
                        help='acc, auc, avg_proba_delta, or median_proba_delta')

    args = parser.parse_args()
    main(args)
