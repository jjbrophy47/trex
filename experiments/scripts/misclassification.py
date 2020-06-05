"""
Explanation of missclassified test instances for the NC17_EvalPart1 (train) and
MFC18_EvalPart1 (test) dataset using TREX. Visualizes the most important feature
from the raw data perspective (positive vs negative), then weighting it using the weights
for a global explanation then weighting it using similarity x abs(weight) for a local explanation.
This also plots the weight distribution, as well as thr similarity vs weight distribution
for a single test instance.
"""
import os
import sys
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner

import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.base import clone
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

import trex
from utility import model_util
from utility import data_util
from utility import print_util
from utility import exp_util


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def _get_top_features(x, shap_vals, feature_list, k=1, normalize=False):
    """
    Parameters
    ----------
    x: 1d array like
        Feature values for this instance.
    shap_vals: 1d array like
        Feature contributions to the prediction.
    feature: 1d array like
        Feature names.
    k: int (default=5)
        Only keep the top k features.

    Returns a list of (feature_name, feature_value, feature_shap) tuples.
    """
    assert len(x) == len(shap_vals) == len(feature_list)
    shap_sort_ndx = np.argsort(np.abs(shap_vals))[::-1]

    if normalize:
        shap_vals /= np.sum(np.abs(shap_vals))

    return list(zip(feature_list[shap_sort_ndx], x[shap_sort_ndx],
                    shap_vals[shap_sort_ndx]))[:k]


def _plot_feature_histograms(args, results, test_val, out_dir):
    """
    Plot the density of the most important feature weighted
    by training instance importance and similarity to the test instance.
    """

    train_feature_vals = results['train_feature_vals']
    train_feature_bins = results['train_feature_bins']
    train_pos_ndx = results['train_pos_ndx']
    train_neg_ndx = results['train_neg_ndx']
    train_weight = results['train_weight']
    train_sim = results['train_sim']
    feature_name = results['target_feature']

    # # plot
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # plot contributions
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
    fig, axs = plt.subplots(1, 3, figsize=(width, height))
    axs = axs.flatten()

    # unweighted
    ax = axs[0]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha, label='positive instances')
    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha, label='negative instances')
    ax.axvline(test_val, color='k', linestyle='--')
    ax.set_xlabel('value')
    ax.set_ylabel('density')
    ax.set_title('Unweighted')
    ax.legend()
    ax.tick_params(axis='both', which='major')

    # weighted by TREX's global weights
    ax = axs[1]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha,
            weights=train_weight[train_pos_ndx])

    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha,
            weights=train_weight[train_neg_ndx])
    ax.axvline(test_val, color='k', linestyle='--')

    ax.set_xlabel('value')
    ax.set_title(r'Weighted by $\alpha$',)
    ax.tick_params(axis='both', which='major')

    # weighted by TREX's global weights * similarity to the test instance
    train_sim_weight = train_weight * train_sim
    ax = axs[2]
    ax.hist(train_feature_vals[train_pos_ndx], bins=train_feature_bins,
            color='g', hatch='.', alpha=args.alpha,
            weights=train_sim_weight[train_pos_ndx])

    ax.hist(train_feature_vals[train_neg_ndx], bins=train_feature_bins,
            color='r', hatch='\\', alpha=args.alpha,
            weights=train_sim_weight[train_neg_ndx])
    ax.axvline(test_val, color='k', linestyle='--')

    ax.set_xlabel('value')
    ax.set_title(r'Weighted by $\alpha * \gamma$')
    ax.tick_params(axis='both', which='major')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '{}.{}'.format(feature_name, args.ext)))


def _plot_instance_histograms(args, logger, results, out_dir):
    """
    Plot TREX's:
      weight distribution and
      similarity * weight distribution.
    """
    train_weight = results['train_weight']
    train_sim = results['train_sim']
    pred_label = results['trex_x_test_pred']

    train_sim_weight = train_sim * train_weight

    s = '[{:15}] mean: {:>12.7f}, median: {:>12.7f}, sum: {:>12.7f}'
    logger.info(s.format('weight', np.mean(train_weight),
                np.median(train_weight), np.sum(train_weight)))
    logger.info(s.format('sim', np.mean(train_sim),
                np.median(train_sim), np.sum(train_sim)))
    logger.info(s.format('sim * weight', np.mean(train_sim_weight),
                np.median(train_sim_weight), np.sum(train_sim_weight)))

    all_vals = np.concatenate([train_weight, train_sim_weight])
    bins = np.histogram(all_vals, bins=args.max_bins)[1]
    bins = None

    if args.coverage:

        # ordering influential samples by biggest absolute influence
        train_sim_weight_sum = np.sum(np.abs(train_sim_weight))
        train_sim_weight_sorted = np.argsort(np.abs(train_sim_weight))[::-1]

        pct_prediction = args.coverage * 100
        logger.info('\nexplaining {}% of the prediction'.format(pct_prediction))

        total = 0
        n_samples = 0
        for i, ndx in enumerate(train_sim_weight_sorted):
            total += np.abs(train_sim_weight[ndx])
            # logger.info('total: {} sum: {}'.format(total, train_sim_weight_sum))
            if total / train_sim_weight_sum >= args.coverage:
                n_samples = i
                break

        if n_samples == 0:
            n_samples = len(train_weight)

        indices = train_sim_weight_sorted[:n_samples]
        pct_data = len(indices) / len(train_weight) * 100

        logger.info('{:.2f}% of the data'.format(pct_data))
        logger.info('no. influential samples: {:,}'.format(len(indices)))

        v1 = train_weight[indices]
        v2 = train_sim_weight[indices]

    else:

        # ordering influential samples by minimum no. needed to explain the predicted label
        train_sim_weight_sum = np.sum(np.abs(train_sim_weight))
        train_sim_weight_neg = np.where(train_sim_weight < 0)[0]
        train_sim_weight_pos = np.where(train_sim_weight > 0)[0]

        if pred_label == 0:
            target = train_sim_weight_pos
            new_ndx = np.argsort(train_sim_weight[train_sim_weight_neg])
            ep = train_sim_weight_neg[new_ndx]
        else:
            target = train_sim_weight_neg
            new_ndx = np.argsort(train_sim_weight[train_sim_weight_pos])[::-1]
            ep = train_sim_weight_pos[new_ndx]

        total = 0
        coverage = 0
        n_ep = 0
        finished = False
        target_sum = np.sum(np.abs(train_sim_weight[target]))

        surplus = 0 if not args.surplus else (train_sim_weight_sum - target_sum) / args.surplus

        for i, ndx in enumerate(ep):
            total += np.abs(train_sim_weight[ndx])
            # logger.info('total: {}, target: {}, sum: {}'.format(total, target_sum, train_sim_weight_sum))
            if total > (target_sum + surplus):
                n_ep = i
                coverage = (total + target_sum) / train_sim_weight_sum
                finished = True
                break

        if finished:
            indices = np.concatenate([target, ep[:n_ep]])
        else:
            logger.info('did not finish')

        pct_data = len(indices) / len(train_weight) * 100
        pct_prediction = coverage * 100
        pct_surplus = surplus / train_sim_weight_sum * 100

        logger.info('no. influential samples: {:,}'.format(len(indices)))
        logger.info('{:.2f}% of the data'.format(pct_data))
        logger.info('{:.2f}% of the prediction'.format(pct_prediction))
        logger.info('+{:.1f}% surplus of the prediction'.format(pct_surplus))

        v1 = train_weight[indices]
        v2 = train_sim_weight[indices]

    # plot contributions
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 4))
    fig = plt.figure(figsize=(width, height * 1.25))

    axs = []
    axs.append(plt.subplot(141))
    axs.append(plt.subplot(142, sharey=axs[0]))
    axs.append(plt.subplot(143))
    axs.append(plt.subplot(144))

    # plot weight distribution for the training samples
    ax = axs[0]
    sns.distplot(v1, color='orange', ax=axs[0], kde=args.kde, bins=bins)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('density')
    ax.tick_params(axis='both', which='major')
    if args.xlim:
        ax.set_xlim(-args.xlim, args.xlim)

    # plot similarity x weight for the training samples
    i = 0 if args.overlay else 1
    ax = axs[i]
    sns.distplot(v2, color='green', ax=ax, kde=args.kde, bins=bins)
    ax.set_xlabel(r'$\alpha * \gamma$')
    ax.set_ylabel('density')
    ax.tick_params(axis='both', which='major')
    if args.xlim:
        ax.set_xlim(-args.xlim, args.xlim)

    plt.tight_layout()

    ax = axs[2]

    # compute density
    xy = np.vstack([train_weight[indices], train_sim[indices]])
    z = gaussian_kde(xy)(xy)
    ax.scatter(train_weight[indices], train_sim[indices], c=z, s=20, edgecolor='')
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.set_ylabel(r'$\gamma$')
    ax.set_xlabel(r'$\alpha$')

    ax = axs[3]
    xy = np.vstack([train_weight[indices], train_sim_weight[indices]])
    z = gaussian_kde(xy)(xy)
    ax.scatter(train_weight[indices], train_sim_weight[indices], c=z, s=20, edgecolor='')
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\alpha * \gamma$')

    ax2 = ax.twinx()

    train_weight_sorted_ndx = np.argsort(train_weight[indices])
    train_weight_sorted = train_weight[indices][train_weight_sorted_ndx]
    train_sim_weight_sorted = train_sim_weight[indices][train_weight_sorted_ndx]
    train_sim_weight_sorted_cumsum = np.cumsum(train_sim_weight_sorted)

    ax2.plot(train_weight_sorted, train_sim_weight_sorted_cumsum,
             label='contribution', color='k', linestyle='--')
    ax2.set_ylabel(r'$\sum \alpha * \gamma$')

    fig.suptitle('Explaining {:.1f}% of the prediction using {:.1f}% of the data'.format(pct_prediction, pct_data))
    plt.tight_layout()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    setting = 'minimum_{:.1f}'.format(pct_surplus) if not args.coverage else '{:.1f}'.format(args.coverage)
    plt.savefig(os.path.join(out_dir, 'plot_{}.{}'.format(setting, args.ext)))


def experiment(args, logger, out_dir, seed):

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=seed)

    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir,
                              return_feature=True,
                              mismatch=True)

    X_train, X_test, y_train, y_test, label, feature = data

    logger.info('train shape: {}'.format(X_train.shape))
    logger.info('test shape: {}'.format(X_test.shape))
    logger.info('no. features: {:,}'.format(len(feature)))

    logger.info('y_train label1: {:,}'.format(np.sum(y_train)))
    logger.info('y_test label1: {:,}'.format(np.sum(y_test)))

    # train a tree ensemble
    logger.info('training the tree ensemble...')
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test, logger=logger)

    # load a TREX model
    model_dir = os.path.join(out_dir, '../models/')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'explainer.pkl')

    print(model_path)
    if args.load_trex and os.path.exists(model_path):
        assert os.path.exists(model_path)
        logger.info('loading TREX model from {}...'.format(model_path))
        explainer = exp_util.load_model(model_path)

    # train TREX
    else:
        logger.info('building TREX...')
        X_val = exp_util.get_val_data(X_train, args.val_frac, seed)
        explainer = trex.TreeExplainer(tree, X_train, y_train,
                                       tree_kernel=args.tree_kernel,
                                       random_state=seed,
                                       kernel_model=args.kernel_model,
                                       kernel_model_kernel=args.kernel_model_kernel,
                                       true_label=args.true_label,
                                       X_val=X_val,
                                       logger=logger)
        logger.info('saving TREX model to {}...'.format(model_path))
        # explainer.save(model_path)

    # extract predictions
    start = time.time()
    logger.info('\ngenerating tree predictions...')
    y_test_pred_tree = tree.predict(X_test)
    logger.info('time: {:.3f}s'.format(time.time() - start))

    start = time.time()
    logger.info('generating tree probabilities...')
    y_test_proba_tree = tree.predict_proba(X_test)[:, 1]
    logger.info('time: {:.3f}s'.format(time.time() - start))

    wrong_label0 = np.where((y_test_pred_tree == 1) & (y_test == 0))[0]
    wrong_label1 = np.where((y_test_pred_tree == 0) & (y_test == 1))[0]
    total_wrong = len(wrong_label0) + len(wrong_label1)
    logger.info('wrong - label0: {}, label1: {}, total: {}'.format(
                len(wrong_label0), len(wrong_label1), total_wrong))

    if args.test_type == 'pos_correct':
        correct_indices = np.where((y_test_pred_tree == 1) & (y_test == 1))[0]
        np.random.seed(seed)
        test_ndx = np.random.choice(correct_indices)

    elif args.test_type == 'neg_correct':
        correct_indices = np.where((y_test_pred_tree == 0) & (y_test == 0))[0]
        np.random.seed(seed)
        test_ndx = np.random.choice(correct_indices)

    # instances where pred=1 but label=0
    elif args.test_type == 'pos_incorrect':
        missed_indices = np.where((y_test_pred_tree == 1) & (y_test == 0))[0]
        np.random.seed(seed)
        test_ndx = np.random.choice(missed_indices)

    # instances where pred=0 but label=1
    elif args.test_type == 'neg_incorrect':
        missed_indices = np.where((y_test_pred_tree == 0) & (y_test == 1))[0]
        np.random.seed(seed)
        test_ndx = np.random.choice(missed_indices)

    else:
        raise ValueError('unknown test_type!')

    # pick an instance where the person is <= 17 from the adult dataset
    indices = np.where(X_test[:, 0] <= 17)[0]
    np.random.seed(seed)
    test_ndx = np.random.choice(indices, size=1)[0]

    x_test = X_test[[test_ndx]]

    # show explanations for missed instances
    logger.info('\ntest index: {}, label: {}'.format(test_ndx, y_test[test_ndx]))
    logger.info('tree pred: {} ({:.3f})'.format(y_test_pred_tree[test_ndx],
                                                y_test_proba_tree[test_ndx]))
    if args.kernel_model == 'lr':
        logger.info('TREX pred: {} ({:.3f})'.format(explainer.predict(x_test)[0],
                                                    explainer.predict_proba(x_test)[:, 1][0]))
    else:
        logger.info('TREX pred: {}'.format(explainer.predict(x_test)[0]))

    # obtain most important features
    shap_explainer = shap.TreeExplainer(tree)
    logger.info('\ncomputing SHAP values for x_test...')
    test_shap = shap_explainer.shap_values(x_test)
    top_features = _get_top_features(x_test[0], test_shap[0], feature,
                                     k=args.topk, normalize=args.normalize_shap)

    # randomly select data to train on
    logger.info('train on a random subset of data')
    np.random.seed(seed)
    indices = np.random.choice(X_train.shape[0], size=int(X_train.shape[0] * args.s_frac), replace=False)
    logger.info('{}: {:,} -> {:,}'.format(args.s_frac, X_train.shape[0], len(indices)))

    new_tree = clone(clf).fit(X_train[indices], y_train[indices])
    model_util.performance(tree, X_train, y_train, X_test, y_test, logger=logger)
    model_util.performance(new_tree, X_train, y_train, X_test, y_test, logger=logger)

    logger.info('\ntraining 2 different LR models')
    l1 = LogisticRegression().fit(X_train, y_train)
    l2 = LogisticRegression().fit(X_train[indices], y_train[indices])
    model_util.performance(l1, X_train, y_train, X_test, y_test, logger=logger)
    model_util.performance(l2, X_train, y_train, X_test, y_test, logger=logger)

    # return

    # for top_feature in top_features:
    #     logger.info(top_feature)

    # get positive and negative training samples
    train_pos_ndx = np.where(y_train == 1)[0]
    train_neg_ndx = np.where(y_train == 0)[0]

    # get global weights and similarity to the test instance
    train_weight = explainer.get_weight()[0]
    sim = explainer.similarity(X_test[[test_ndx]])[0]
    contributions = train_weight * sim

    if False:

        # retrain to find gold standard sample impacts
        samples = 250
        np.random.seed(seed)
        indices = np.random.choice(X_train.shape[0], size=samples, replace=False)
        impacts = []

        x_test_label = y_test[test_ndx]
        proba = tree.predict_proba(x_test)[:, 1]

        for i in tqdm(indices):
            new_X_train = np.delete(X_train, i, axis=0)
            new_y_train = np.delete(y_train, i)
            new_tree = clone(clf).fit(new_X_train, new_y_train)
            new_proba = new_tree.predict_proba(x_test)[:, 1]
            diff = proba - new_proba
            if x_test_label == 0:
                impact = diff
            elif x_test_label == 1:
                impact = diff * -1
            impacts.append(impact)

        fig, ax = plt.subplots()
        ax.scatter(contributions[indices], impacts)
        plt.show()

    # # training on the biggest weighted training samples
    # sort_ndx = np.argsort(np.abs(train_weight))[::-1]
    # indices = sort_ndx[:50]

    # logger.info('training on top training samples')
    # new_tree = clone(clf).fit(X_train[indices], y_train[indices],
    #                           sample_weight=np.abs(train_weight[indices]))
    # model_util.performance(tree, X_train, y_train, X_test, y_test, logger=logger)
    # model_util.performance(new_tree, X_train, y_train, X_test, y_test, logger=logger)

    if False:

        np.random.seed(seed)
        X_test_sub_ndx = np.random.choice(X_test.shape[0], size=int(X_test.shape[0] * 0.25))
        X_test_sub = X_test[X_test_sub_ndx]
        y_test_sub = y_test[X_test_sub_ndx]

        # get net contributions of training samples on test data
        contributions_list = []
        for i in tqdm(range(X_test_sub.shape[0])):
            x_test_sim = explainer.similarity(X_test[[i]])[0]
            x_test_contributions = train_weight * x_test_sim
            contributions_list.append(x_test_contributions)
        contributions_arr = np.vstack(contributions_list)
        contributions_sum = np.sum(contributions_arr, axis=0)
        # np.save(os.path.join(model_dir, 'contributions_sum'), contributions_sum)

        contributions_sum_total = np.sum(contributions_sum)
        sort_ndx = np.argsort(np.abs(contributions_sum))[::-1]
        total = 0
        indices = []
        for ndx in sort_ndx:
            total += contributions_sum[ndx]
            indices.append(ndx)
            if total / contributions_sum_total >= args.max_contribution:
                break

        indices = sort_ndx[:500]

        # # random baseline
        # np.random.seed(seed)
        # indices = np.random.choice(X_train.shape[0], size=len(indices), replace=False)

        # logger.info('{}: {}'.format(len(indices), sorted(indices)))

        # nonzero_weight_ndx = np.where(np.abs(contributions_sum) > 0)[0]
        # zero_weight_ndx = np.setdiff1d(np.arange(len(train_weight)), nonzero_weight_ndx)
        # print(contributions_sum, len(contributions_sum))
        # print(contributions_sum[indices], len(contributions_sum[indices]))

        # train a new tree ensemble weighted using the weighted training samples
        new_tree = clone(clf).fit(X_train[indices], y_train[indices],
                                  sample_weight=np.abs(contributions_sum[indices]))
        # new_tree = clone(clf).fit(X_train[indices], y_train[indices])
        new_proba = new_tree.predict_proba(X_test_sub)[:, 1]
        old_proba = tree.predict_proba(X_test_sub)[:, 1]

        pcorr = pearsonr(new_proba, old_proba)[0]
        scorr = spearmanr(new_proba, old_proba)[0]
        logger.info('pearson: {:.3f}, spearman: {:.3f}'.format(pcorr, scorr))

        model_util.performance(tree, X_train, y_train, X_test_sub, y_test_sub, logger=logger)
        model_util.performance(new_tree, X_train[indices], y_train[indices],
                               X_test_sub, y_test_sub, logger=logger)

        fig, ax = plt.subplots()
        ax.scatter(new_proba, old_proba, color='purple')
        ax.set_ylabel('Original GBDT probability')
        ax.set_xlabel('New GBDT probability')
        ax.set_title('Dataset: {}\nTraining samples: {:,} -> {:,}\npearson: {:.3f}, spearman: {:.3f}'.format(
                     args.dataset, len(X_train), len(indices), pcorr, scorr))
        plt.show()

        # random baseline
        np.random.seed(seed)
        indices = np.random.choice(X_train.shape[0], size=len(indices), replace=False)

        # logger.info('{}: {}'.format(len(indices), sorted(indices)))

        # nonzero_weight_ndx = np.where(np.abs(contributions_sum) > 0)[0]
        # zero_weight_ndx = np.setdiff1d(np.arange(len(train_weight)), nonzero_weight_ndx)
        # print(contributions_sum, len(contributions_sum))
        # print(contributions_sum[indices], len(contributions_sum[indices]))

        # train a new tree ensemble weighted using the weighted training samples
        # new_tree = clone(clf).fit(X_train[indices], y_train[indices],
        #                           sample_weight=np.abs(contributions_sum[indices]))
        new_tree = clone(clf).fit(X_train[indices], y_train[indices])
        new_proba = new_tree.predict_proba(X_test_sub)[:, 1]
        old_proba = tree.predict_proba(X_test_sub)[:, 1]

        pcorr = pearsonr(new_proba, old_proba)[0]
        scorr = spearmanr(new_proba, old_proba)[0]
        logger.info('pearson: {:.3f}, spearman: {:.3f}'.format(pcorr, scorr))

        model_util.performance(tree, X_train, y_train, X_test_sub, y_test_sub, logger=logger)
        model_util.performance(new_tree, X_train[indices], y_train[indices],
                               X_test_sub, y_test_sub, logger=logger)

        fig, ax = plt.subplots()
        ax.scatter(new_proba, old_proba, color='purple')
        ax.set_ylabel('Original GBDT probability')
        ax.set_xlabel('New GBDT probability')
        ax.set_title('Dataset: {}\nTraining samples: {:,} -> {:,}\npearson: {:.3f}, spearman: {:.3f}'.format(
                     args.dataset, len(X_train), len(indices), pcorr, scorr))
        plt.show()

    # weight_sort_ndx = np.argsort(train_weight)
    # step_size = int(len(train_weight) / 20)

    # # remove training samples in batches
    # fig, ax = plt.subplots()

    # scores = []
    # pct_removed = []
    # logger.info('removing samples from smallest to biggest...')
    # for i in range(0, len(train_weight) - step_size, step_size):
    #     keep_indices = weight_sort_ndx[i:]
    #     new_X_train = X_train[keep_indices]
    #     new_y_train = y_train[keep_indices]
    #     new_tree = clf.fit(new_X_train, new_y_train)
    #     if np.sum(new_y_train) > 0 and np.sum(new_y_train) < len(new_y_train):
    #         scores.append(accuracy_score(y_test, new_tree.predict(X_test)))
    #         pct_removed.append(i / len(train_weight) * 100)

    # ax.plot(pct_removed, scores, marker='.', label='small to big')

    # weight_sort_ndx = weight_sort_ndx[::-1]
    # scores = []
    # pct_removed = []
    # logger.info('removing samples from biggest to smallest...')
    # for i in range(0, len(train_weight) - step_size, step_size):
    #     keep_indices = weight_sort_ndx[i:]
    #     new_X_train = X_train[keep_indices]
    #     new_y_train = y_train[keep_indices]
    #     new_tree = clf.fit(new_X_train, new_y_train)
    #     if np.sum(new_y_train) > 0 and np.sum(new_y_train) < len(new_y_train):
    #         scores.append(accuracy_score(y_test, new_tree.predict(X_test)))
    #         pct_removed.append(i / len(train_weight) * 100)

    # ax.plot(pct_removed, scores, marker='.', label='big to small')

    # ax.legend()
    # plt.show()

    logger.info('\ncollecting results...')
    results = explainer.get_params()
    results['test_ndx'] = test_ndx
    results['train_pos_ndx'] = train_pos_ndx
    results['train_neg_ndx'] = train_neg_ndx
    results['train_weight'] = train_weight
    results['train_sim'] = sim
    results['trex_x_test_pred'] = explainer.predict(x_test)[0]

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=6)

    # shap values for train and test
    X_train_shap_fp = os.path.join(model_dir, 'X_train_shap.npy')
    X_test_shap_fp = os.path.join(model_dir, 'X_test_shap.npy')

    if os.path.exists(X_train_shap_fp):
        X_train_shap = np.load(X_train_shap_fp)
    else:
        logger.info('\ncomputing SHAP values for X_train...')
        X_train_shap = shap_explainer.shap_values(X_train)
        np.save(X_train_shap_fp, X_train_shap)

    if os.path.exists(X_test_shap_fp):
        X_test_shap = np.load(X_test_shap_fp)
    else:
        logger.info('computing SHAP values for X_test...')
        X_test_shap = shap_explainer.shap_values(X_test)
        np.save(X_test_shap_fp, X_test_shap)

    # shap.summary_plot(X_train_shap, X_train, feature_names=feature)
    # shap.summary_plot(X_test_shap, X_test, feature_names=feature)

    # feature-based explanation for test instance
    logger.info('test {}'.format(test_ndx))
    logger.info('{}'.format(dict(zip(feature, X_test[test_ndx]))))
    shap.summary_plot(test_shap, x_test, feature_names=feature)

    # pos_target = np.where((X_train[:, 0] <= 17) & (y_train == 1))[0]
    # neg_target = np.where((X_train[:, 0] <= 17) & (y_train == 0))[0]

    contributions_sum = np.sum(np.abs(contributions))
    # indices = np.argsort(np.abs(contributions))[::-1]
    indices = np.argsort(contributions)[::-1]

    # # feature-based explanations for the most influential training instances
    # for ndx in pos_target[:5]:
    #     w = train_weight[ndx]
    #     s = sim[ndx]
    #     c = contributions[ndx] / contributions_sum
    #     logger.info('train {}, weight: {}, sim: {}, contribution (normalized): {}'.format(ndx, w, s, c))
    #     logger.info('{}'.format(dict(zip(feature, X_train[ndx]))))
    #     shap.summary_plot(X_train_shap[[ndx]], X_train[[ndx]], feature_names=feature)

    # # feature-based explanations for the most influential training instances
    # for ndx in neg_target[:5]:
    #     w = train_weight[ndx]
    #     s = sim[ndx]
    #     c = contributions[ndx] / contributions_sum
    #     logger.info('train {}, weight: {}, sim: {}, contribution (normalized): {}'.format(ndx, w, s, c))
    #     logger.info('{}'.format(dict(zip(feature, X_train[ndx]))))
    #     shap.summary_plot(X_train_shap[[ndx]], X_train[[ndx]], feature_names=feature)

    # feature-based explanations for the most influential training instances
    for ndx in indices[:args.topk]:
        w = train_weight[ndx]
        s = sim[ndx]
        c = contributions[ndx] / contributions_sum
        my_str = 'train {}, label: {}, weight: {}, sim: {}, contribution (normalized): {}'
        logger.info(my_str.format(ndx, y_train[ndx], w, s, c))
        for name, val in list(zip(feature, X_train[ndx])):
            logger.info('  {}: {}'.format(name, val))
        # logger.info('{}'.format(dict(zip(feature, X_train[ndx]))))
        # shap.summary_plot(X_train_shap[[ndx]], X_train[[ndx]], feature_names=feature)

    # explain features
    for target_feature, test_val, shap_val in top_features:
        feat_ndx = np.where(target_feature == feature)[0][0]

        train_feature_bins = np.histogram(X_train[:, feat_ndx], bins=args.max_bins)[1]

        results['target_feature'] = target_feature
        results['train_feature_vals'] = X_train[:, feat_ndx]
        results['train_feature_bins'] = train_feature_bins

        _plot_feature_histograms(args, results, test_val, out_dir)

    _plot_instance_histograms(args, logger, results, out_dir)
    np.save(os.path.join(out_dir, 'results.npy'), results)


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, args.tree_type,
                           args.tree_kernel, 'rs{}'.format(args.rs))
    os.makedirs(out_dir, exist_ok=True)

    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)
    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/misclassification/', help='output directory.')

    parser.add_argument('--val_frac', type=float, default=0.05, help='amount of training data to use for validation.')

    parser.add_argument('--max_bins', type=int, default=40, help='number of bins for feature values.')
    parser.add_argument('--alpha', type=float, default=0.5, help='transparency value.')
    parser.add_argument('--xlim', type=float, default=None, help='x limits on instance plots.')
    parser.add_argument('--kde', action='store_true', default=None, help='plot kde on weight distribution.')
    parser.add_argument('--overlay', action='store_true', default=None, help='overlay weight distributions.')
    parser.add_argument('--topk', type=int, default=1, help='number of features to show.')
    parser.add_argument('--test_type', type=str, default='correct', help='instance to try and explain.')
    parser.add_argument('--load_trex', action='store_true', default=False, help='load TREX.')
    parser.add_argument('--normalize_shap', action='store_true', default=False, help='normalize SHAP values.')
    parser.add_argument('--coverage', type=float, default=0.9, help='fraction of contributions to explain.')
    parser.add_argument('--surplus', type=float, default=None, help='multiplier for surplus minimum contributions.')
    parser.add_argument('--max_contribution', type=float, default=None, help='contribution level.')

    parser.add_argument('--tree_type', type=str, default='lgb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth.')

    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='kernel model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='similarity kernel')
    parser.add_argument('--true_label', action='store_true', default=False, help='train TREX on the true labels.')

    parser.add_argument('--ext', type=str, default='png', help='output image format.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)


class Args:
    dataset = 'nc17_mfc18'
    data_dir = 'data'
    out_dir = 'output/misclassification/'

    val_frac = 0.05

    max_bins = 40
    alpha = 0.5
    xlim = 0.05
    kde = False
    overlay = False
    topk = 1
    test_type = 'correct'
    load_trex = False
    normalize_shap = False
    coverage = 0.1
    surplus = None
    max_contribution = 0.9

    tree_type = 'lgb'
    n_estimators = 100
    max_depth = None

    tree_kernel = 'leaf_output'
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'
    true_label = False

    ext = 'pdf'

    rs = 1
    verbose = 0
