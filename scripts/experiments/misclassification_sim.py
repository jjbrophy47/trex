"""
Explanation of missclassified test instances.
"""
import os
import sys
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from scipy.interpolate import interpn
from matplotlib.colors import Normalize
from matplotlib import cm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for trex
import trex
import util


def density_scatter(x, y, fig, ax, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    assert fig is not None
    assert ax is not None

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]),
                0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


def show_instances(args, X_train, alpha, sim, attribution, y_train, indices, k, logger):
    """
    Display train instance weight, similarity, attribution, label and `age` value
    ordered by `indices`.
    """
    s = '[{:5,}: {:5}] label: {}, alpha: {:.3f}, sim: {:.3f} attribution: {:.3f}, age: {:.0f}'
    for i, ndx in enumerate(indices[:k]):
        logger.info(s.format(i, ndx, y_train[ndx], alpha[ndx], sim[ndx],
                    attribution[ndx], X_train[ndx][args.age_ndx]))


def get_height(width, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return height


def plot_similarity(args, alpha, sim, out_dir, logger):
    """
    Plot similarity of training instances against their instance weights.
    """

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('axes', labelsize=11)
    plt.rc('axes', titlesize=11)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=6)

    width = 4.8  # Machine Learning journal
    height = get_height(width=width, subplots=(1, 1))
    fig, ax = plt.subplots(figsize=(width * 1.65, height * 1.25))

    density_scatter(alpha, sim, fig, ax, bins=args.n_bins, rasterized=args.rasterize)
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_ylabel(r'Similarity ($\gamma$)')
    ax.set_xlabel(r'Weight ($\alpha$)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'similarity.pdf'), bbox_inches='tight', dpi=200)
    logger.info('saving plot to {}/...'.format(os.path.join(out_dir)))


def experiment(args, logger, out_dir):

    # start timer
    begin = time.time()

    # create random number generator
    rng = np.random.default_rng(args.rs)

    # get data
    data = util.get_data(args.dataset,
                         data_dir=args.data_dir,
                         preprocessing=args.preprocessing,
                         mismatch=True)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # get tree-ensemble
    clf = util.get_model(args.model,
                         n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         random_state=args.rs,
                         cat_indices=cat_indices)

    # display dataset statistics
    logger.info('\nno. train instances: {:,}'.format(X_train.shape[0]))
    logger.info('no. test instances: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))
    logger.info('\npos. label % (train): {:.1f}%'.format(np.sum(y_train) / y_train.shape[0] * 100))
    logger.info('pos. label % (test): {:.1f}%\n'.format(np.sum(y_test) / y_test.shape[0] * 100))

    # train tree ensemble
    model = clone(clf).fit(X_train, y_train)
    util.performance(model, X_train, y_train, logger=logger, name='Train')
    util.performance(model, X_test, y_test, logger=logger, name='Test')

    # train surrogate model
    params = {'C': args.C, 'n_neighbors': args.n_neighbors, 'tree_kernel': args.tree_kernel}
    surrogate = trex.train_surrogate(model=model,
                                     surrogate='klr',
                                     X_train=X_train,
                                     y_train=y_train,
                                     val_frac=args.tune_frac,
                                     metric=args.metric,
                                     seed=args.rs,
                                     params=params,
                                     logger=logger)

    # extract predictions
    start = time.time()
    model_pred = model.predict(X_test)
    model_proba = model.predict_proba(X_test)[:, 1]
    logger.info('predicting...{:.3f}s'.format(time.time() - start))

    # pick a test instance in which the person is <= 17 from the Adult dataset
    indices = np.where(X_test[:, args.age_ndx] <= 17)[0]
    test_ndx = rng.choice(indices)
    age_test_val = X_test[test_ndx][args.age_ndx]
    x_test = X_test[[test_ndx]]

    # show prediction for this test instance
    s = '\ntest: {}, actual: {}, proba.: {:.3f}, age: {:.0f}'
    logger.info(s.format(test_ndx, y_test[test_ndx], model_proba[test_ndx], age_test_val))

    # sort based on similarity-influence
    if 'sim' in args.surrogate:

        # compute influence based on predicted labels
        attributions = surrogate.similarity(x_test)
        pred_label = model.predict(x_test)

        # put positive weight if similar instances have the same label as the predicted test label
        for i in range(x_test.shape[0]):
            attributions[i] = np.where(y_train == pred_label[i], attributions[i], attributions[i] * -1)
        attributions = attributions.sum(axis=0)

        attribution_indices = np.argsort(attributions)[::-1]

    else:
        exit(0)

    # sort training instances by most similar to the test instance
    sim = surrogate.similarity(x_test)[0]
    sim_indices = np.argsort(sim)[::-1]

    # get instance weights
    alpha = surrogate.get_alpha()

    # 1. show most excitatory training instances
    logger.info('\nTop {:,} most excitatory instances to the predicted label...'.format(args.topk_inf))
    show_instances(args, X_train, alpha, sim, attributions, y_train, attribution_indices, args.topk_inf, logger)

    # 1. show most inhibitory training instances
    logger.info('\nTop {:,} most inhibitory instances to the predicted label...'.format(args.topk_inf))
    show_instances(args, X_train, alpha, sim, attributions, y_train, np.argsort(attributions), args.topk_inf, logger)

    # 2. compute change in predicted probability after REMOVING the most influential instances
    s = 'test: {}, actual: {}, proba.: {:.3f}, age: {:.0f}'
    logger.info('\nRemoving top {:,} influential instances..'.format(args.topk_inf))
    new_X_train = np.delete(X_train, attribution_indices[:args.topk_inf], axis=0)
    new_y_train = np.delete(y_train, attribution_indices[:args.topk_inf])
    new_model = clone(clf).fit(new_X_train, new_y_train)
    util.performance(model, new_X_train, new_y_train, logger=logger, name='Train')
    util.performance(model, X_test, y_test, logger=logger, name='Test')
    logger.info(s.format(test_ndx, y_test[test_ndx], new_model.predict_proba(X_test)[:, 1][test_ndx], age_test_val))

    # 3. compute change in predicted probability after FLIPPING the labels of the most influential instances
    s1 = 'test: {}, actual: {}, proba.: {:.3f}, age: {:.0f}'
    s2 = '\n{:,} out of the top {:,} most (excitatory) influential instances have age <= 17'
    logger.info('\nFixing ONLY corrupted labels of the top {:,} influential instances..'.format(args.topk_inf))
    new_X_train = X_train.copy()
    new_y_train = y_train.copy()

    # fix a portion of the corrupted training instances
    temp_indices = np.where(X_train[attribution_indices][:args.topk_inf][:, args.age_ndx] <= 17)[0]
    age17_topk_inf_indices = attribution_indices[temp_indices]
    new_y_train[age17_topk_inf_indices] = 0

    # fit new model and re-evaluate
    new_model = clone(clf).fit(new_X_train, new_y_train)
    util.performance(new_model, new_X_train, new_y_train, logger=logger, name='Train')
    util.performance(new_model, X_test, y_test, logger=logger, name='Test')
    logger.info(s1.format(test_ndx, y_test[test_ndx], new_model.predict_proba(X_test)[:, 1][test_ndx], age_test_val))
    logger.info(s2.format(age17_topk_inf_indices.shape[0], args.topk_inf))

    # 6. of the most influential train instances (pos. and neg.), compute how many have age <= 17
    neg_inf_indices = np.argsort(attributions)
    neg_inf_num_age17 = np.where(X_train[neg_inf_indices][:args.topk_inf][:, args.age_ndx] <= 17)[0].shape[0]
    logger.info('\nTop {:,} neg. influence instances with age <= 17: {:,}'.format(args.topk_inf, neg_inf_num_age17))

    # 6. of the most influential train instances (pos. and neg.), compute how many have age <= 17
    abs_inf_indices = np.argsort(np.abs(attributions))[::-1]
    abs_inf_num_age17 = np.where(X_train[abs_inf_indices][:args.topk_inf][:, args.age_ndx] <= 17)[0].shape[0]
    logger.info('\nTop {:,} abs. influence instances with age <= 17: {:,}'.format(args.topk_inf, abs_inf_num_age17))

    # 4. show most similar training instances
    logger.info('\nTop {:,} most similar samples to the predicted label...'.format(args.topk_sim))
    show_instances(args, X_train, alpha, sim, attributions, y_train, sim_indices, args.topk_sim, logger)

    # 6. of the most similar train instances, compute how many have age <= 17
    num_age17_topk = np.where(X_train[sim_indices][:args.topk_sim][:, args.age_ndx] <= 17)[0].shape[0]
    logger.info('\nTop {:,} most similar train instances with age <= 17: {:,}'.format(args.topk_sim, num_age17_topk))

    # 7. plot similarity of train instances against their instance weights
    logger.info('\nplotting similarity vs. weights...')
    plot_similarity(args, alpha, sim, out_dir, logger)

    # 8. no. train instances with age <= 17 and an alpha coefficient < 0
    neg_alpha_indices = np.where(attributions < 0)[0]
    num_age17_neg_alpha = np.where(X_train[neg_alpha_indices][:, args.age_ndx] <= 17)[0].shape[0]
    logger.info('\nno. instances with age <= 17 and alpha < 0: {:,}'.format(num_age17_neg_alpha))

    # 9. no. train instances with age <= 17 and an alpha coefficient >= 0
    pos_alpha_indices = np.where(attributions >= 0)[0]
    num_age17_pos_alpha = np.where(X_train[pos_alpha_indices][:, args.age_ndx] <= 17)[0].shape[0]
    logger.info('no. instances with age <= 17 and alpha >= 0: {:,}'.format(num_age17_pos_alpha))

    # 10. no. train instances with age <= 17, similarity > thershold and an alpha coefficient < 0
    s = 'no. instances with age <= 17, sim > {:.2f} and alpha >= 0: {:,}'
    neg_alpha_indices = np.where((attributions < 0) & (sim > args.sim_thresh))[0]
    num_age17_sim_neg_alpha = np.where(X_train[neg_alpha_indices][:, args.age_ndx] <= 17)[0].shape[0]
    logger.info(s.format(args.sim_thresh, num_age17_sim_neg_alpha))

    # 11. no. train instances with age <= 17, similarity > thershold and an alpha coefficient >= 0
    s = 'no. instances with age <= 17, sim > {:.2f} and alpha >= 0: {:,}'
    pos_alpha_indices = np.where((attributions >= 0) & (sim > args.sim_thresh))[0]
    num_age17_sim_pos_alpha = np.where(X_train[pos_alpha_indices][:, args.age_ndx] <= 17)[0].shape[0]
    logger.info(s.format(args.sim_thresh, num_age17_sim_pos_alpha))

    # display total time
    logger.info('\ntotal time: {:.3f}s'.format(time.time() - begin))


def main(args):

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           'rs_{}'.format(args.rs))

    # create output directory
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset to explain.')
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/misclassification/', help='output directory.')

    # Experiment settings
    parser.add_argument('--age_ndx', type=int, default=0, help='index of the `age` attribute.')
    parser.add_argument('--sim_thresh', type=float, default=0.75, help='similairity threshold.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')

    # Explanation settings
    parser.add_argument('--topk_inf', type=int, default=5, help='no. top influential train instances to show.')
    parser.add_argument('--topk_sim', type=int, default=400, help='no. most similar train instances to show.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=250, help='number of trees.')
    parser.add_argument('--max_depth', type=int, default=5, help='maximum depth.')

    # Surrogate settings
    parser.add_argument('--tune_frac', type=float, default=0.0, help='amount of data for validation.')
    parser.add_argument('--surrogate', type=str, default='klr_sim', help='surrogate model.')
    parser.add_argument('--C', type=float, default=1.0, help='penalty parameters for KLR or SVM.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='no. neighbors to use for KNN.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_path', help='tree kernel.')
    parser.add_argument('--metric', type=str, default='mse', help='fidelity metric.')

    # Plot settings
    parser.add_argument('--n_bins', type=int, default=50, help='number of bins similarity plot.')
    parser.add_argument('--rasterize', action='store_true', default=False, help='rasterize dense instance plots.')

    args = parser.parse_args()
    main(args)
