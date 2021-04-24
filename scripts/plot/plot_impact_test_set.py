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
    dataset_list = ['churn', 'surgical', 'vaccine', 'bank_marketing', 'adult', 'synthetic']

    # label, color, marker, linestyle, zorder
    methods = {}

    # slice 'n dice TREX results
    if args.kernel is None or args.kernel == 'to':
        if args.trex_type is None or args.trex_type == 'alpha':
            methods['klr_og_tree_output_alpha'] = ['TREX (OG-TO-Alpha)', 'yellowgreen', '2', '-', 11]
        if args.trex_type is None or args.trex_type == 'sim':
            methods['klr_og_tree_output_sim'] = ['TREX (OG-TO-Sim)', 'yellowgreen', '2', '--', 11]
        if args.trex_type is None or args.trex_type == 'alphasim':
            methods['klr_og_tree_output'] = ['TREX (OG-TO-AlphaSim)', 'yellowgreen', '2', ':', 11]

    if args.kernel is None or args.kernel == 'lp':
        if args.trex_type is None or args.trex_type == 'alpha':
            methods['klr_og_leaf_path_alpha'] = ['TREX (OG-LP-Alpha)', 'cyan', '1', '-', 11]
        if args.trex_type is None or args.trex_type == 'sim':
            methods['klr_og_leaf_path_sim'] = ['TREX (OG-LP-Sim)', 'cyan', '1', '--', 10]
        if args.trex_type is None or args.trex_type == 'alphasim':
            methods['klr_og_leaf_path'] = ['TREX (OG-LP-AlphaSim)', 'cyan', '1', ':', 11]

    if args.kernel is None or args.kernel == 'wlp':
        if args.trex_type is None or args.trex_type == 'alpha':
            methods['klr_og_weighted_leaf_path_alpha'] = ['TREX (OG-WLP-Alpha)', 'magenta', '2', '-', 11]
        if args.trex_type is None or args.trex_type == 'sim':
            methods['klr_og_weighted_leaf_path_sim'] = ['TREX (OG-WLP-Sim)', 'magenta', '2', '--', 11]
        if args.trex_type is None or args.trex_type == 'alphasim':
            methods['klr_og_weighted_leaf_path'] = ['TREX (OG-WLP-AlphaSim)', 'magenta', '2', ':', 11]

    methods['klr_og_weighted_feature_path_sim_C-1.0'] = ['TREX (OG-WFP-AlphaSim)', 'gold', '2', '--', 11]

    methods['random'] = ['Random', 'red', 'o', '-', 9]
    # methods['maple+'] = ['MAPLE', 'orange', '^', '--', 7]
    # methods['maple+_og'] = ['MAPLE (OG)', 'orange', '^', ':', 7]
    # methods['leaf_influence'] = ['LeafInfluence', 'black', '.', '-', 1]
    methods['fast_leaf_influence'] = ['FastLeafInfluence', 'black', '.', '--', 1]
    # methods['knn_og'] = ['TEKNN (OG)', 'magenta', 'h', '--', 5]
    # methods['bacon'] = ['Bacon', 'brown', '2', '--', 5]

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['model'] == args.model]
    df = df[df['setting'] == args.setting]

    # plot settings
    util.plot_settings(fontsize=13, markersize=5)

    # inches
    width = 4.8  # Machine Learning journal
    height = util.get_height(width=width, subplots=(2, 3))

    for train_frac_to_remove in args.train_frac_to_remove:
        qf = df[df['train_frac_to_remove'] == train_frac_to_remove]

        fig, axs = plt.subplots(2, 3, figsize=(width * 3, height * 5))
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
                temp_df1 = qf[qf['dataset'] == dataset]

                # add y-axis
                if j == 0:

                    if args.metric == 'proba_diff':
                        ax.set_ylabel(r'|Test prob. $\Delta$|')

                    elif args.metric == 'proba':
                        ax.set_ylabel(r'Test prob.')

                    elif args.metric == 'avg_loss':
                        ax.set_ylabel(r'L1 loss')

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

                    n_runs = temp_df2['num_runs'].values[0]

                    # extract performance results
                    temp_df2 = temp_df2.iloc[0]
                    metric_mean = temp_df2['{}_mean'.format(args.metric)]
                    metric_sem = temp_df2['{}_sem'.format(args.metric)]
                    removed_pcts = temp_df2['remove_pct']

                    # convert results from strings to arrays
                    metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
                    metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
                    removed_pcts = np.fromstring(removed_pcts[1: -1], dtype=np.float32, sep=' ')

                    # TEMP
                    yerr = metric_sem
                    ax_label = 'n={:,}'.format(n_runs)
                    line = ax.errorbar(removed_pcts, metric_mean, yerr=yerr, marker=marker,
                                       linestyle=linestyle, color=color, zorder=zorder, label=ax_label)
                    ax.legend(fontsize=6)

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
                               args.metric)
        os.makedirs(out_dir, exist_ok=True)

        # adjust legend
        fig.legend(tuple(lines), tuple(labels), loc='center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.1))
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.265, wspace=0.3)

        # save figure
        fp = os.path.join(out_dir, '{}.pdf'.format(train_frac_to_remove))
        plt.savefig(fp)
        print('saving to {}...'.format(fp))

        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/impact_test_set/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/impact_test_set/', help='output directory.')
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model.')
    parser.add_argument('--metric', type=str, default='acc', help='peformance metric.')
    parser.add_argument('--setting', type=str, default='static', help='evaluation setting.')

    parser.add_argument('--train_frac_to_remove', type=float, nargs='+', default=[0.1, 0.25, 0.5], help='fracion.')

    # filter settings
    parser.add_argument('--C', type=float, default=None, help='specific C.')
    parser.add_argument('--trex_type', type=str, default=None, help='None, alpha, or sim.')
    parser.add_argument('--kernel', type=str, default=None, help='None, to, lp, or wlp.')

    args = parser.parse_args()
    main(args)
