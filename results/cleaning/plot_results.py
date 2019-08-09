"""
This script takes results from multiople datasets, and plots them vertically.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _dataset_results(dataset):

    acc_dir = os.path.join(dataset, 'effectiveness')
    eff_dir = os.path.join(dataset, 'efficiency')

    res = {}
    method_list = ['sexee', 'rand', 'tree_loss', 'svm_loss', 'influence']

    # read in results
    for method in method_list:
        res[method] = {}
        res[method]['check_pct'] = np.load(os.path.join(acc_dir, '{}_check_pct.npy'.format(method)))
        res[method]['acc'] = np.load(os.path.join(acc_dir, '{}_acc.npy'.format(method)))
        res[method]['fix_pct'] = np.load(os.path.join(eff_dir, '{}_fix_pct.npy'.format(method)))
    res['acc_test_clean'] = np.load(os.path.join(acc_dir, 'acc_test_clean.npy'))

    return res


def _plot_graph(ax, res, y_key, xlabel=None, ylabel=None):

    inf_label = 'leaf_inf (all)'

    l1 = ax.plot(res['sexee']['check_pct'], res['sexee'][y_key], marker='.', color='g', label='ours')
    l2 = ax.plot(res['rand']['check_pct'], res['rand'][y_key], marker='^', color='r', label='random')
    l3 = ax.plot(res['tree_loss']['check_pct'], res['tree_loss'][y_key], marker='p', color='c', label='tree_loss')
    l4 = ax.plot(res['svm_loss']['check_pct'], res['svm_loss'][y_key], marker='*', color='y', label='svm_loss')
    l5 = ax.plot(res['influence']['check_pct'], res['influence'][y_key], marker='+', color='m', label=inf_label)
    ax.tick_params(axis='both', which='major', labelsize=24)

    if y_key == 'acc':
        ax.axhline(res['acc_test_clean'], color='k', linestyle='--')

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=24)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=24)

    return l1[0], l2[0], l3[0], l4[0], l5[0]


def main():

    res_adult = _dataset_results('adult')
    res_amazon = _dataset_results('amazon')

    fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex='col')
    axs = axs.flatten()
    l1, l2, l3, l4, l5 = _plot_graph(axs[0], res_adult, 'acc', xlabel=None, ylabel='test accuracy')
    _plot_graph(axs[1], res_amazon, 'acc', xlabel=None, ylabel=None)
    _plot_graph(axs[2], res_adult, 'fix_pct', 'fraction of train data checked', 'fraction of flips fixed')
    _plot_graph(axs[3], res_amazon, 'fix_pct', 'fraction of train data checked', ylabel=None)
    axs[0].set_title('Adult', fontsize=28)
    axs[1].set_title('Amazon', fontsize=28)

    fig.legend((l1, l2, l3, l4, l5), ('ours', 'random', 'tree_loss', 'svm_loss', 'leaf_inf (all)'),
               loc='center', ncol=3, bbox_to_anchor=(0.5, 0.048), fontsize=24)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig('cleaning.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
