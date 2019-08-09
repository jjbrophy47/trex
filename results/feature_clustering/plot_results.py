"""
This script plots the data before and after feature transformations.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _get_results(dataset):

    res = {}
    res['train_pos'] = np.load(os.path.join(dataset, 'train_positive.npy'))
    res['train_neg'] = np.load(os.path.join(dataset, 'train_negative.npy'))
    res['test_pos'] = np.load(os.path.join(dataset, 'test_positive.npy'))
    res['test_neg'] = np.load(os.path.join(dataset, 'test_negative.npy'))
    return res


def _plot_graph(ax, res, xlabel=None, ylabel=None, alpha=0.5):

    ax.scatter(res['train_neg'][:, 0], res['train_neg'][:, 1], color='blue', alpha=alpha, label='train (y=0)',
               marker='+')
    ax.scatter(res['train_pos'][:, 0], res['train_pos'][:, 1], color='red', alpha=alpha, label='train (y=1)',
               marker='1')
    ax.scatter(res['test_neg'][:, 0], res['test_neg'][:, 1], color='cyan', alpha=alpha, label='test (y=0)',
               marker='x')
    ax.scatter(res['test_pos'][:, 0], res['test_pos'][:, 1], color='orange', alpha=alpha, label='test (y=1)',
               marker='2')

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def main():

    # read in results
    res_nc17none = _get_results('NC17_EvalPart1_none')
    res_nc17leafoutput = _get_results('NC17_EvalPart1_leaf_output')
    res_nc17mfc18leafoutput = _get_results('nc17_mfc18_leaf_output')

    # plot results
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(12, 9))

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])  # second row, using all columns

    _plot_graph(ax0, res_nc17none, xlabel='tsne 0', ylabel='tsne 1')
    _plot_graph(ax1, res_nc17leafoutput, xlabel='tsne 0', ylabel=None)
    _plot_graph(ax2, res_nc17mfc18leafoutput, xlabel='tsne 0', ylabel='tsne 1')

    ax2.legend(loc='right', bbox_to_anchor=(0.75, -0.25), ncol=4)
    fig.subplots_adjust(bottom=0.2)

    plt.savefig('clustering.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
