"""
This script plots the cleaning experiment results.
"""
import os
import sys
import argparse
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=RuntimeWarning)  # true divide

import numpy as np
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util


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
    dataset_list = ['churn', 'surgical', 'vaccine', 'bank_marketing', 'adult']

    methods = {}

    if args.kernel is None or args.kernel == 'to':
        if args.trex_type is None or args.trex_type == 'alpha':
            methods['klr_og_tree_output_alpha'] = ['TREX (OG-TO-Alpha)', 'yellowgreen', '2', '-', 11]
        if args.trex_type is None or args.trex_type == 'sim':
            methods['klr_og_tree_output_sim'] = ['TREX (OG-TO-Sim)', 'yellowgreen', '2', '--', 11]

    if args.kernel is None or args.kernel == 'lp':
        if args.trex_type is None or args.trex_type == 'alpha':
            methods['klr_og_leaf_path_alpha'] = ['TREX (OG-LP-Alpha)', 'cyan', '1', '-', 11]
        if args.trex_type is None or args.trex_type == 'sim':
            methods['klr_og_leaf_path_sim'] = ['TREX (OG-LP-Sim)', 'cyan', '1', '--', 10]

    if args.kernel is None or args.kernel == 'wlp':
        if args.trex_type is None or args.trex_type == 'alpha':
            methods['klr_og_weighted_leaf_path_alpha'] = ['TREX (OG-WLP-Alpha)', 'magenta', '2', '-', 11]
        if args.trex_type is None or args.trex_type == 'sim':
            methods['klr_og_weighted_leaf_path_sim'] = ['TREX (OG-WLP-Sim)', 'magenta', '2', '--', 11]

    # methods['klr'] = ['TREX', 'blue', '1', '-', 11]  # label, color, marker, linestyle, zorder
    methods['random'] = ['Random', 'red', 'o', '-', 9]
    methods['tree_loss'] = ['GBDT Loss', 'green', 'v', '-', 3]
    # methods['klr_loss'] = ['KLR Loss', 'purple', '^', '-', 2]
    # methods['maple'] = ['MAPLE', 'orange', '>', '-', 7]
    # methods['leaf_influence'] = ['LeafInfluence', 'black', '.', '-', 1]
    methods['fast_leaf_influence'] = ['FastLeafInfluence', 'black', '.', '--', 1]
    # methods['tree_prototype'] = ['TreeProto', '#EEC64F', '*', '-', 6]
    # methods['knn'] = ['TEKNN', 'yellow', 'h', '-', 5]
    # methods['knn_loss'] = ['KNN Loss', 'brown', 's', '-', 8]
    # methods['klr_og'] = ['TREX OG', 'cyan', '.', '-', 11]
    # methods['klr_loss_og'] = ['KLR Loss OG', 'magenta', '.', '-', 10]

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['model'] == args.model]

    # matplotlib settings
    util.plot_settings(fontsize=13)

    # inches
    # width = 4.8  # Machine Learning journal
    # height = get_height(width=width, subplots=(2, 3))
    # fig, axs = plt.subplots(2, 3, figsize=(width * 1.75, height * 2.25))
    # fig, axs = plt.subplots(2, 3, figsize=(width * 1.75, height * 2.5))

    for flip_frac in args.flip_frac:
        qf = df[df['flip_frac'] == flip_frac]

        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        lines = []
        labels = []

        # dataset incrementer
        k = 0

        # plot datasets
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):

                if k == 5:
                    break

                # extract dataset results
                ax = axs[i][j]
                dataset = dataset_list[k]
                temp_df1 = qf[qf['dataset'] == dataset]

                # # obtain clean results
                line_clean = None
                if args.metric in ['acc', 'auc']:
                    metric_clean = temp_df1['{}_clean'.format(args.metric)].mean()
                    line_clean = ax.axhline(metric_clean, color='k', linestyle='--')

                # add y-axis
                if j == 0:
                    if args.metric in ['acc', 'auc']:
                        label = 'accuracy' if args.metric == 'acc' else 'AUC'
                        ax.set_ylabel('Test {}'.format(label))

                    elif args.metric == 'fixed_pct':
                        ax.set_ylabel('Corrupted labels fixed (%)')

                # add x-axis
                if i == 1:
                    ax.set_xlabel('Train data checked (%)')

                # add title
                ax.set_title('Census (10%)' if dataset == 'census_0p1' else dataset.capitalize())
                ax.tick_params(axis='both', which='major')

                # plot each method
                for method, (label, color, marker, linestyle, zorder) in methods.items():

                    # get method results
                    temp_df2 = temp_df1[temp_df1['method'] == method]

                    if len(temp_df2) == 0:
                        print('[Missing] {}: {}'.format(dataset, method))
                        continue

                    # extract performance results
                    temp_df2 = temp_df2.iloc[0]
                    metric_mean = temp_df2['{}s_mean'.format(args.metric)]
                    metric_sem = temp_df2['{}s_sem'.format(args.metric)]
                    checked_pcts = temp_df2['checked_pcts']

                    # convert results from strings to arrays
                    metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
                    metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
                    checked_pcts = np.fromstring(checked_pcts[1: -1], dtype=np.float32, sep=' ')

                    # plot
                    line = ax.errorbar(checked_pcts, metric_mean, yerr=metric_sem, marker=marker,
                                       linestyle=linestyle, color=color, zorder=zorder)

                    # save for legend
                    if i == 0 and j == 0:
                        lines.append(line)
                        labels.append(label)

                # add reference line
                if i == 0 and j == 0 and line_clean is not None:
                    lines.append(line_clean)
                    labels.append('Clean')

                # increment dataset
                k += 1

        # create output directory
        out_dir = os.path.join(args.out_dir, args.model, args.metric)
        os.makedirs(out_dir, exist_ok=True)

        # adjust legend
        fig.legend(tuple(lines), tuple(labels), loc='center', ncol=4, bbox_to_anchor=(0.5, 0.0925))

        # adjust figure
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.3, wspace=0.3)

        # save figure
        fp = os.path.join(out_dir, 'flip_{}.pdf'.format(flip_frac))
        plt.savefig(fp)
        print('saving to {}...'.format(fp))

        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_dir', type=str, default='output/cleaning_new/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/cleaning_new/', help='output directory.')

    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble.')
    parser.add_argument('--flip_frac', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4], help='flips.')
    parser.add_argument('--metric', type=str, default='acc', help='acc, auc, or fixed_pcts.')

    parser.add_argument('--trex_type', type=str, default=None, help='None, alpha, or sim.')
    parser.add_argument('--kernel', type=str, default=None, help='None, to, lp, or wlp.')

    parser.add_argument('--rs', type=int, nargs='+', default=list(range(1, 11)), help='random state.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
