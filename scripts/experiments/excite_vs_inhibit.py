"""
Experiment:
    1) Select an ambiguously predicted test instance (i.e. predicted prob. around 0.5)
    2a) Remove most excitatory training instances in equal increments.
    2b) Remove most inhibitory trainnig instances in equal increments.
    3) Train a new tree-ensemble on each reduced training set.
    4) Record changes in predicted probability.

Removing the most excitatory training instances should drive the predicted probability to 0.
Removing the most inhibitory training instances should drive the predicted probability to 1.
"""
import os
import sys
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility
import trex
import util


def get_height(width, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return height


def measure_performance(args, train_indices, clf, X_train, y_train, X_test, y_test,
                        logger=None):
    """
    Measures the change in predictions as training instances are removed.
    """

    # baseline predicted probability
    model = clone(clf).fit(X_train, y_train)
    base_proba = model.predict_proba(X_test)[:, 1]

    # display status
    if logger:
        logger.info('test label: {}, before prob.: {:.5f}'.format(int(y_test[0]), base_proba[0]))

    # result container
    result = {}
    result['proba'] = [base_proba[0]]
    result['remove_pct'] = [0]

    # compute no. samples to remove between each checkpoint
    n_checkpoint = int(X_train.shape[0] * args.train_frac_to_remove / args.n_checkpoints)

    # remove percentages of training samples and retrain
    for i in range(args.n_checkpoints):
        start = time.time()

        # compute how many samples should be removed
        n_remove = (i + 1) * n_checkpoint
        remove_indices = train_indices[:n_remove]
        pct_remove = n_remove / X_train.shape[0] * 100

        # remove most influential training samples
        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        # measure change in test instance probability
        new_model = clone(clf).fit(new_X_train, new_y_train)
        proba = new_model.predict_proba(X_test)[:, 1]
        proba_diff = np.abs(base_proba - proba)[0]

        # add to results
        result['proba'].append(proba[0])
        result['remove_pct'].append(pct_remove)

        # display progress
        if logger:
            s = '[{:.1f}% removed] after prob.: {:.5f}, delta: {:.5f}...{:.3f}s'
            logger.info(s.format(n_remove / X_train.shape[0] * 100, proba[0], proba_diff, time.time() - start))

    return result


def random_method(X_train, rng):
    """
    Randomly orders the training intances to be removed.
    """
    return rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)


def trex_method(args, model, X_train, y_train, X_test, logger=None,
                frac_progress_update=0.1):
    """
    Sort training instances by most excitatory or most inhibitory on the test set.
    """

    # train surrogate model
    params = {'C': args.C, 'n_neighbors': args.n_neighbors, 'tree_kernel': args.tree_kernel}
    surrogate = trex.train_surrogate(model=model,
                                     surrogate=args.method,
                                     X_train=X_train,
                                     y_train=y_train,
                                     val_frac=args.tune_frac,
                                     metric=args.metric,
                                     seed=args.rs,
                                     params=params,
                                     logger=logger)

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # sort instances by most excitatory or most inhibitory
    attributions = surrogate.compute_attributions(X_test)
    attributions_sum = np.sum(attributions, axis=0)

    # sort by most excitatory and inhibitory
    n_excitatory = len(np.where(attributions_sum > 0)[0])
    n_inhibitory = len(np.where(attributions_sum < 0)[0])
    excitatory_train_indices = np.argsort(attributions_sum)[::-1][:n_excitatory]
    inhibitory_train_indices = np.argsort(attributions_sum)[:n_inhibitory]

    # display attributions from the top k train instances
    if logger:
        k = 20

        # display most excitatory training instances
        sim_s = surrogate.similarity(X_test)[0][excitatory_train_indices]
        alpha_s = surrogate.get_alpha()[excitatory_train_indices]
        attributions_sum_s = attributions_sum[excitatory_train_indices]
        y_train_s = y_train[excitatory_train_indices]

        train_info = list(zip(excitatory_train_indices, attributions_sum_s, y_train_s, alpha_s, sim_s))
        s = '[{:5}] label: {}, alpha: {:.3f}, sim: {:.3f} attribution sum: {:.3f}'

        logger.info('\nExcitatory training instances...')
        for ndx, atr, lab, alpha, sim in train_info[:k]:
            logger.info(s.format(ndx, lab, alpha, sim, atr))

        # display most inhibitory training instances
        sim_s = surrogate.similarity(X_test)[0][inhibitory_train_indices]
        alpha_s = surrogate.get_alpha()[inhibitory_train_indices]
        attributions_sum_s = attributions_sum[inhibitory_train_indices]
        y_train_s = y_train[inhibitory_train_indices]

        train_info = list(zip(inhibitory_train_indices, attributions_sum_s, y_train_s, alpha_s, sim_s))
        s = '[{:5}] label: {}, alpha: {:.3f}, sim: {:.3f} attribution sum: {:.3f}'

        logger.info('\nInhibitory training instances...')
        for ndx, atr, lab, alpha, sim in train_info[:k]:
            logger.info(s.format(ndx, lab, alpha, sim, atr))

    return excitatory_train_indices, inhibitory_train_indices


def sort_train_instances(args, model, X_train, y_train, X_test, y_test, rng,
                         order='excitatory', logger=None):
    """
    Sorts training instance to be removed using one of several methods.
    """

    # random method
    if args.method == 'random':
        train_indices = random_method(X_train, rng)

    # TREX method
    elif 'klr' in args.method or 'svm' in args.method:
        train_indices = trex_method(args, model, X_train, y_train, X_test, logger=logger)

    else:
        raise ValueError('method {} unknown!'.format(args.method))

    return train_indices


def experiment(args, logger, out_dir):
    """
    Main method that removes training instances ordered by
    different methods and measure their impact on a random
    set of test instances.
    """

    # start timer
    begin = time.time()

    # create random number generator
    rng = np.random.default_rng(args.rs)

    # get data
    data = util.get_data(args.dataset,
                         data_dir=args.data_dir,
                         preprocessing=args.preprocessing)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # get tree-ensemble
    clf = util.get_model(args.model,
                         n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         random_state=args.rs,
                         cat_indices=cat_indices)

    # use a fraction of the train data
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train_samples = int(X_train.shape[0] * args.train_frac)
        train_indices = rng.choice(X_train.shape[0], size=n_train_samples, replace=False)
        X_train, y_train = X_train[train_indices], y_train[train_indices]

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    util.performance(model, X_train, y_train, logger=logger, name='Train')

    # select an ambiguously predicted test instance
    proba = model.predict_proba(X_test)[:, 1]
    sorted_indices = np.argsort(np.abs(proba - 0.5))
    test_indices = sorted_indices[:1]  # shape=(1,)
    X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

    # display dataset statistics
    logger.info('\nno. train instances: {:,}'.format(X_train.shape[0]))
    logger.info('no. test instances: {:,}'.format(X_test_sub.shape[0]))
    logger.info('no. features: {:,}\n'.format(X_train.shape[1]))
    logger.info('pos. label % (test): {:.1f}%\n'.format(np.sum(y_test) / y_test.shape[0] * 100))

    # sort train instances
    exc_indices, inh_indices = trex_method(args, model, X_train, y_train, X_test_sub, logger=logger)
    ran_indices = rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)
    ran_pos_indices = np.where(y_train == 1)[0]
    ran_neg_indices = np.where(y_train == 0)[0]
    rng.shuffle(ran_pos_indices)
    rng.shuffle(ran_neg_indices)

    # remove, retrain, and re-evaluate
    logger.info('\nremoving most excitatory train instances...')
    exc_result = measure_performance(args, exc_indices, clf, X_train, y_train, X_test_sub, y_test_sub, logger=logger)

    logger.info('\nremoving most inhibitory train instances...')
    inh_result = measure_performance(args, inh_indices, clf, X_train, y_train, X_test_sub, y_test_sub, logger=logger)

    logger.info('\nremoving train instances uniformly at random...')
    ran_result = measure_performance(args, ran_indices, clf, X_train, y_train, X_test_sub, y_test_sub, logger=logger)

    if args.extra_methods:
        logger.info('\nremoving positive train instances at random...')
        ran_pos_result = measure_performance(args, ran_pos_indices, clf, X_train, y_train, X_test_sub, y_test_sub,
                                             logger=logger)

        logger.info('\nremoving negative train instances at random...')
        ran_neg_result = measure_performance(args, ran_neg_indices, clf, X_train, y_train, X_test_sub, y_test_sub,
                                             logger=logger)

    # matplotlib settings
    util.plot_settings(fontsize=13)

    # inches
    width = 4.8  # Machine Learning journal
    height = get_height(width=width, subplots=(1, 1))
    fig, ax = plt.subplots(figsize=(width * 1.65, height * 1.0))

    # plot results
    l1 = ax.errorbar(exc_result['remove_pct'], exc_result['proba'], color='blue', linestyle='--',
                     marker='.', label='Most excitatory')
    l2 = ax.errorbar(inh_result['remove_pct'], inh_result['proba'], color='green', linestyle='--',
                     marker='+', label='Most inhibitory')
    l3 = ax.errorbar(ran_result['remove_pct'], ran_result['proba'], color='red', linestyle='-',
                     marker='*', label='Random')
    lines = [l1, l2, l3]
    labels = ['Most excitatory', 'Most inhibitory', 'Random']

    if args.extra_methods:
        l4 = ax.errorbar(ran_pos_result['remove_pct'], ran_pos_result['proba'], color='cyan', linestyle=':',
                         marker='1', label='Pos. random')
        l5 = ax.errorbar(ran_neg_result['remove_pct'], ran_neg_result['proba'], color='orange', linestyle=':',
                         marker='2', label='Neg. random')
        lines += [l4, l5]
        labels += ['Random (pos. only)', 'Random (neg. only)']

    ax.set_xlabel('Train data removed (%)')
    ax.set_ylabel('Predicted probability')
    ax.set_ylim(0, 1)

    # adjust legend
    fig.legend(tuple(lines), tuple(labels), loc='left', ncol=1,
               bbox_to_anchor=(1.0, 0.85), title='Removal Ordering')
    plt.tight_layout()
    fig.subplots_adjust(right=0.65)

    # save plot
    plt.savefig(os.path.join(out_dir, 'probas.pdf'), bbox_inches='tight')

    # display results
    logger.info('\nsaving results to {}/...'.format(os.path.join(out_dir)))
    logger.info('total time: {:.3f}s'.format(time.time() - begin))


def main(args):

    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.preprocessing,
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
    parser.add_argument('--dataset', type=str, default='vaccine', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/excite_vs_inhibit/', help='directory to save results.')

    # Data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='fraction of train data to evaluate.')
    parser.add_argument('--tune_frac', type=float, default=0.0, help='amount of data for validation.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=250, help='no. of trees.')
    parser.add_argument('--max_depth', type=int, default=5, help='max. depth in tree ensemble.')

    # Method settings
    parser.add_argument('--method', type=str, default='klr', help='method.')
    parser.add_argument('--extra_methods', action='store_true', default=True, help='random pos. and random neg.')
    parser.add_argument('--metric', type=str, default='mse', help='metric for tuning surrogate models.')

    # No tuning settings
    parser.add_argument('--C', type=float, default=1.0, help='penalty parameters for KLR or SVM.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='no. neighbors to use for KNN.')
    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='tree kernel.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--n_test', type=int, default=1, help='no. of test instances to evaluate.')
    parser.add_argument('--train_frac_to_remove', type=float, default=0.1, help='fraction of train data to remove.')
    parser.add_argument('--n_checkpoints', type=int, default=10, help='no. checkpoints to perform retraining.')

    # Additional settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
