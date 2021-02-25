"""
Experiment:
    1) Generate an instance-attribution explanation for a set of test instances.
    2) Sort training instances by influence on the selected test instnces.
    3) Remove a fixed number of most influential training instances.
    4) Retrain a new tree ensemble on the remaining dataset.
    5) Evaluate predictive performance on the test set.

If the instances removed are highly influential, than performance should decrease,
or the average change in test prediction should be significant.
"""
import os
import sys
import time
import uuid
import resource
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
from copy import deepcopy
from datetime import datetime

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility
import trex
from utility import model_util
from utility import data_util
from utility import print_util
from baselines.influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from baselines.maple.MAPLE import MAPLE


def score(model, X_test, y_test):
    """
    Evaluates the model the on test set and returns metric scores.
    """

    # 1 test sample
    if y_test.shape[0] == 1:
        result = (-1, -1)

    # >1 test sample
    else:
        acc = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return (acc, auc)

    return result


def measure_performance(train_indices, n_checkpoint, n_checkpoints,
                        clf, X_train, y_train, X_test, y_test, logger=None):
    """
    Measures the change in predictions as training instances are removed.
    """

    # compute baseline statistics
    model = clone(clf).fit(X_train, y_train)
    acc, auc = score(model, X_test, y_test)
    base_proba = model.predict_proba(X_test)[:, 1]

    # result container
    result = {}
    result['accs'] = [acc]
    result['aucs'] = [auc]
    result['avg_proba_deltas'] = [0]
    result['median_proba_deltas'] = [0]
    result['removed_pcts'] = [0]

    # result containers
    start = time.time()
    s = '[Checkpoint] removed: {:.1f}%; Acc.: {:.3f}; AUC: {:.3f}'
    s += '; Proba. delta: {:.3f} (avg.), {:.3f} (median); cum. time: {:.3f}s'

    # display status
    if logger:
        logger.info('\nremoving and re-evaluting...')

    # remove percentages of training samples and retrain
    for i in range(n_checkpoints):

        # compute how many samples should be removed
        n_samples = (i + 1) * n_checkpoint
        remove_indices = train_indices[:n_samples]

        # remove most influential training samples
        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        # only samples from one class remain
        if len(np.unique(new_y_train)) == 1:
            raise ValueError('Only samples from one class remain!')

        # remeasure test instance predictive performance
        new_model = clone(clf).fit(new_X_train, new_y_train)
        acc, auc = score(new_model, X_test, y_test)
        proba = new_model.predict_proba(X_test)[:, 1]

        # add to results
        result['accs'].append(acc)
        result['aucs'].append(auc)
        result['avg_proba_deltas'].append(np.mean(np.abs(base_proba - proba)))
        result['median_proba_deltas'].append(np.median(np.abs(base_proba - proba)))
        result['removed_pcts'].append(n_samples / train_indices.shape[0] * 100)

        # display progress
        if logger:
            logger.info(s.format(result['removed_pcts'][-1], result['accs'][-1], result['aucs'][-1],
                                 result['avg_proba_deltas'][-1], result['median_proba_deltas'][-1],
                                 time.time() - start))

    return result


def random_method(X_train, rng):
    """
    Randomly orders the training intances to be removed.
    """
    return rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)


def trex_method(args, model, X_train, y_train, X_test, logger=None,
                frac_progress_update=0.1):
    """
    Sort training instances by largest 'excitatory' influnce on the test set.
    """

    # train surrogate model
    kernel_model = args.method.split('-')[0].split('_')[0]
    tree_kernel = args.method.split('-')[-1]
    surrogate = trex.TreeExplainer(model,
                                   X_train,
                                   y_train,
                                   kernel_model=kernel_model,
                                   tree_kernel=tree_kernel,
                                   val_frac=args.tune_frac,
                                   metric=args.metric,
                                   random_state=args.rs,
                                   logger=logger)

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # sort instances with highest positive influence first
    start = time.time()
    attributions_sum = np.zeros(X_train.shape[0])

    # compute impact of each training sample on the test set
    for i in range(X_test.shape[0]):
        attributions_sum += surrogate.compute_attributions(X_test[[i]])[0]

        # display progress
        if logger and i % int(X_test.shape[0] * frac_progress_update) == 0:
            elapsed = time.time() - start
            logger.info('finished {:.1f}% test instances...{:.3f}s'.format((i / X_test.shape[0]) * 100, elapsed))

    # sort instances by their contributions
    train_indices = np.argsort(attributions_sum)[::-1]

    return train_indices


def maple_method(args, model, X_train, y_train, X_test, logger=None,
                 frac_progress_update=0.1):
    """
    Sort training instances using MAPLE's "local training distribution".
    """

    # train a MAPLE explainer model
    train_label = model.predict(X_train)
    maple_explainer = MAPLE(X_train, train_label, X_train, train_label,
                            verbose=args.verbose, dstump=False)

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # contributions container
    start = time.time()
    contributions_sum = np.zeros(X_train.shape[0])

    # compute similarity of each training instance to the set set
    for i in range(X_test.shape[0]):
        contributions = maple_explainer.get_weights(X_test[i])
        contributions_sum += contributions

        # display progress
        if logger and i % int(X_test.shape[0] * frac_progress_update) == 0:
            elapsed = time.time() - start
            logger.info('finished {:.1f}% test instances...{:.3f}s'.format((i / X_test.shape[0]) * 100, elapsed))

    # sort training instances based on similarity to the test set
    train_indices = np.argsort(contributions_sum)[::-1]

    return train_indices


def influence_method(args, model, X_train, y_train, X_test, y_test, logger=None,
                     k=-1, frac_progress_update=0.1):
    """
    Sort training instances based on their Leaf Influence on the test set.

    Reference:
    https://github.com/kohpangwei/influence-release/blob/master/influence/experiments.py
    """
    assert k == -1, 'AllPoints method not used for k: {}'.format(k)
    assert args.model == 'cb', 'tree-ensemble is not a CatBoost model!'

    # save CatBoost model
    temp_fp = '.{}_cb.json'.format(str(uuid.uuid4()))
    model.save_model(temp_fp, format='json')

    # initialize Leaf Influence
    explainer = CBLeafInfluenceEnsemble(temp_fp, X_train, y_train, k=k,
                                        learning_rate=model.learning_rate_,
                                        update_set='AllPoints')

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # contributions container
    start = time.time()
    contributions_sum = np.zeros(X_train.shape[0])

    # compute influence on each test instance
    for i in range(X_test.shape[0]):

        contributions = []
        buf = deepcopy(explainer)

        # compute influence for each training instance
        for j in range(X_train.shape[0]):
            explainer.fit(removed_point_idx=j, destination_model=buf)
            contributions.append(buf.loss_derivative(X_test[[i]], y_test[[i]])[0])

            # display progress
            if logger and j % int(X_train.shape[0] * frac_progress_update) == 0:
                elapsed = time.time() - start
                train_frac_complete = j / X_train.shape * 100
                logger.info('[Test {}] train {:.1f}%...{:.3f}s'.format(i, train_frac_complete, elapsed))

        contributions = np.array(contributions)
        contributions_sum += contributions

    # sort by descending order; the most positive train instances
    # are the ones that decrease the log loss the most, and are the most helpful
    train_indices = np.argsort(contributions_sum)[::-1]

    # clean up
    os.system('rm {}'.format(temp_fp))

    return train_indices


def teknn_method(args, model, X_train, y_train, X_test, logger=None,
                 frac_progress_update=0.1):
    """
    Sort trainnig instance based on similarity density to the test instances.
    """

    # transform the data
    tree_kernel = args.method.split('-')[-1]
    extractor = trex.TreeExtractor(model, tree_kernel=tree_kernel)
    X_train_alt = extractor.fit_transform(X_train)

    # train surrogate model
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]}
    surrogate = trex.util.train_surrogate(model,
                                          'knn',
                                          param_grid,
                                          X_train,
                                          X_train_alt,
                                          y_train,
                                          val_frac=args.tune_frac,
                                          metric=args.metric,
                                          seed=args.rs,
                                          logger=logger)

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # contributions container
    start = time.time()
    contributions_sum = np.zeros(X_train.shape[0])

    # compute the contribution of all training samples on each test instance
    for i in range(X_test.shape[0]):
        x_test_alt = extractor.transform(X_test[[i]])
        pred_label = int(surrogate.predict(x_test_alt)[0])
        distances, neighbor_ids = surrogate.kneighbors(x_test_alt)

        # add density to training instances that are in the neighborhood of the test instance
        for neighbor_id in neighbor_ids[0]:
            contribution = 1 if y_train[neighbor_id] == pred_label else -1
            contributions_sum[neighbor_id] += contribution

        # display progress
        if logger and i % int(X_test.shape[0] * frac_progress_update) == 0:
            elapsed = time.time() - start
            logger.info('finished {:.1f}% test instances...{:.3f}s'.format((i / X_test.shape[0]) * 100, elapsed))

    # sort instances based on similarity density
    train_indices = np.argsort(contributions_sum)[::-1]

    return train_indices


def sort_train_instances(args, model, X_train, y_train, X_test, y_test, rng, logger=None):
    """
    Sorts training instance to be removed using one of several methods.
    """

    # random method
    if args.method == 'random':
        train_indices = random_method(X_train, rng)

    # TREX method
    elif 'klr' in args.method or 'svm' in args.method:
        train_indices = trex_method(args, model, X_train, y_train, X_test, logger=logger)

    # MAPLE
    elif args.method == 'maple':
        train_indices = maple_method(args, model, X_train, y_train, X_test, logger=logger)

    # Leaf Influence (NOTE: can only compute influence of the LOSS, requires label)
    elif args.method == 'leaf_influence':
        train_indices = influence_method(args, model, X_train, y_train, X_test, y_test, logger=logger)

    # TEKNN
    elif 'knn' in args.method:
        train_indices = teknn_method(args, model, X_train, y_train, X_test, logger=logger)

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
    data = data_util.get_data(args.dataset,
                              data_dir=args.data_dir,
                              processing_dir=args.processing_dir)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    # get tree-ensemble
    clf = model_util.get_model(args.model,
                               n_estimators=args.n_estimators,
                               max_depth=args.max_depth,
                               random_state=args.rs,
                               cat_indices=cat_indices)

    # use a fraction of the train data
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train_samples = int(X_train.shape[0] * args.train_frac)
        train_indices = rng.choice(X_train.shape[0], size=n_train_samples, replace=False)
        X_train, y_train = X_train[train_indices], y_train[train_indices]

    # select a subset of test instances uniformly at random
    test_indices = rng.choice(X_test.shape[0], size=args.n_test, replace=False)
    X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

    # choose new subset if test subset all contain the same label
    if args.n_test > 1:
        while y_test_sub.sum() == len(y_test_sub) or y_test_sub.sum() == 0:
            test_indices = rng.choice(X_test.shape[0], size=args.n_test, replace=False)
            X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

    # display dataset statistics
    logger.info('\nno. train instances: {:,}'.format(X_train.shape[0]))
    logger.info('no. test instances: {:,}'.format(X_test_sub.shape[0]))
    logger.info('no. features: {:,}\n'.format(X_train.shape[1]))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, logger=logger, name='Train')
    model_util.performance(model, X_test_sub, y_test_sub, logger=logger, name='Test')

    # compute how many samples to remove before a checkpoint
    if args.train_frac_to_remove >= 1.0:
        n_checkpoint = int(args.train_frac_to_remove)
    elif args.train_frac_to_remove > 0:
        n_checkpoint = int(args.train_frac_to_remove * X_train.shape[0])
    else:
        raise ValueError('invalid train_frac_to_remove: {}'.format(args.train_frac_to_remove))

    # sort train instances, then remove, retrain, and re-evaluate
    train_indices = sort_train_instances(args, model, X_train, y_train, X_test_sub, y_test_sub, rng, logger=logger)
    result = measure_performance(train_indices, n_checkpoint, args.n_checkpoints,
                                 clf, X_train, y_train, X_test_sub, y_test_sub, logger=logger)

    # save results
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['total_time'] = time.time() - begin
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # display results
    logger.info('\nResults:\n{}'.format(result))
    logger.info('\nsaving results to {}...'.format(os.path.join(out_dir, 'results.npy')))


def main(args):

    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.method,
                           args.scoring,
                           'rs_{}'.format(args.rs))

    # create output directory
    os.makedirs(out_dir, exist_ok=True)
    print_util.clear_dir(out_dir)

    # create logger
    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # I/O settings
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--processing_dir', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/roar/', help='directory to save results.')

    # Data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='fraction of train data to evaluate.')
    parser.add_argument('--tune_frac', type=float, default=0.1, help='amount of data for validation.')
    parser.add_argument('--scoring', type=str, default='accuracy', help='metric for tuning.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=10, help='no. of trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='max. depth in tree ensemble.')

    # Method settings
    parser.add_argument('--method', type=str, default='klr-leaf_output', help='method.')
    parser.add_argument('--metric', type=str, default='mse', help='metric for tuning surrogate models.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--n_test', type=int, default=50, help='no. of test instances.')
    parser.add_argument('--train_frac_to_remove', type=float, default=1.0, help='fraction of train data to remove.')
    parser.add_argument('--n_checkpoints', type=int, default=10, help='no. checkpoints to perform retraining.')

    # Additional settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
