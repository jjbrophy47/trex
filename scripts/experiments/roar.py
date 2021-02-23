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
from utility import exp_util
from baselines.influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from baselines.maple.MAPLE import MAPLE


def score(model, X_test, y_test):
    """
    Evaluates the model the on test set and returns metric scores.
    """
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return acc, auc


def measure_performance(train_indices, n_checkpoint, n_checkpoints,
                        X_test, y_test, X_train, y_train, clf):
    """
    Measures the change in log loss as training instances are removed.
    """
    model = clone(clf).fit(X_train, y_train)
    acc, auc = score(model, X_test, y_test)
    base_proba = model.predict_proba(X_test)[:, 1]

    # result container
    result = {}
    result['accs'] = [acc]
    result['aucs'] = [auc]
    result['avg_proba_delta'] = [0]

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
        result['avg_proba_delta'] = np.abs(base_proba - proba).mean()
        result['median_proba_delta'] = np.abs(base_proba - proba).mean()

    return result


def trex_method(args, tree, X_test, X_train, y_train, seed, logger):

    trex_explainer = trex.TreeExplainer(tree, X_train, y_train,
                                        tree_kernel=args.tree_kernel,
                                        random_state=seed,
                                        true_label=args.true_label,
                                        kernel_model=args.kernel_model,
                                        verbose=args.verbose,
                                        val_frac=args.val_frac,
                                        logger=logger)

    # sort instances with highest positive influence first
    contributions_sum = np.zeros(X_train.shape[0])

    train_weight = trex_explainer.get_weight()[0]
    for i in tqdm.tqdm(range(X_test.shape[0])):
        train_sim = trex_explainer.similarity(X_test[[i]])[0]
        contributions = train_weight * train_sim
        contributions_sum += contributions

    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def maple_method(X_test, args, model, X_train, y_train, logger):

    train_label = y_train if args.true_label else model.predict(X_train)

    maple_explainer = MAPLE(X_train, train_label, X_train, train_label,
                            verbose=args.verbose, dstump=False)

    # order the training instances
    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = maple_explainer.get_weights(X_test[i])
        contributions_sum += contributions
    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def influence_method(X_test, args, model, X_train, y_train, y_test, logger):

    model_path = '.model.json'
    model.save_model(model_path, format='json')

    if args.inf_k == -1:
        update_set = 'AllPoints'
    elif args.inf_k == 0:
        update_set = 'SinglePoint'
    else:
        update_set = 'TopKLeaves'

    explainer = CBLeafInfluenceEnsemble(model_path, X_train, y_train, k=args.inf_k,
                                        learning_rate=model.learning_rate_,
                                        update_set=update_set)

    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = []
        buf = deepcopy(explainer)

        for j in tqdm.tqdm(range(len(X_train))):
            explainer.fit(removed_point_idx=j, destination_model=buf)
            contributions.append(buf.loss_derivative(X_test[[i]], y_test[[i]])[0])

        contributions = np.array(contributions)
        contributions_sum += contributions

    # sort by descending order; the most positive train instances
    # are the ones that decrease the log loss the most, and are the most helpful
    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def teknn_method(args, model, X_test, X_train, y_train, y_test, seed, logger):

    global teknn_explainer
    global teknn_extractor

    if teknn_explainer is None:

        # transform the data
        teknn_extractor = trex.TreeExtractor(model, tree_kernel=args.tree_kernel)
        X_train_alt = teknn_extractor.fit_transform(X_train)
        train_label = y_train if args.true_label else model.predict(X_train)

        # tune and train teknn
        teknn_explainer = exp_util.tune_knn(model, X_train, X_train_alt, train_label, args.val_frac,
                                            seed=1, logger=logger)

    # results container
    contributions_sum = np.zeros(X_train.shape[0])

    # compute the contribution of all training samples on each test instance
    for i in tqdm.tqdm(range(X_test.shape[0])):
        x_test_alt = teknn_extractor.transform(X_test[[i]])
        pred_label = int(teknn_explainer.predict(x_test_alt)[0])
        distances, neighbor_ids = teknn_explainer.kneighbors(x_test_alt)

        for neighbor_id in neighbor_ids[0]:
            contribution = 1 if y_train[neighbor_id] == pred_label else -1
            contributions_sum[neighbor_id] += contribution

    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def experiment(args, logger, out_dir, seed):
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
    while y_test_sub.sum() == len(y_test_sub) or y_test_sub.sum() == 0:
        test_indices = rng.choice(X_test.shape[0], size=args.n_test, replace=False)
        X_test_sub, y_test_sub = X_test[test_indices], y_test[test_indices]

    # display dataset statistics
    logger.info('no. train instances: {:,}'.format(len(X_train)))
    logger.info('no. test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, logger=logger, name='Train')
    model_util.performance(model, X_test_sub, y_test_sub, logger=logger, name='Test')

    # pcts = list(range(0, 100, 10))

    # compute how many samples to remove before a checkpoint
    n_checkpoint = (args.frac_train_to_remove * X_train.shape[0]) / args.n_checkpoints

    # np.save(os.path.join(out_dir, 'percentages.npy'), pcts)

    # random method
    if args.method == 'random':
        train_indices = rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)

    # TREX method
    elif 'klr' in args.method or 'svm' in args.method:
        train_indices = trex_method(args, model, X_test, X_train, y_train, seed, logger)

    # MAPLE
    elif args.method == 'maple':
        train_indices = maple_method(X_test, args, model, X_train, y_train, logger)

    # Leaf Influence
    elif args.method == 'leaf_influence':
        train_indices = influence_method(X_test, args, model, X_train, y_train, y_test, logger)

    # TEKNN
    elif args.method == 'knn':
        train_indices = teknn_method(args, model, X_test, X_train, y_train, y_test, seed, logger)

    else:
        raise ValueError('method {} unknown!'.format(args.method))

    # remove and retrain
    result = measure_performance(train_indices, n_checkpoint, X_test, y_test, X_train, y_train, clf)

    # save rsults
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['total_time'] = time.time() - begin
    np.save(os.path.join(out_dir, 'results.npy'), result)


def main(args):

    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.method,
                           args.scoring,
                           'rs{}'.format(args.rs))

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
    parser.add_argument('--processing_dir', type=str, default='sandard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/roar/', help='directory to save results.')

    # Data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--tune_frac', type=float, default=0.1, help='Amount of data for validation.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')

    # Method settings
    parser.add_argument('--method', type=str, default='klr-leaf_output', help='method.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--n_test', type=int, default=50, help='number of test instances.')
    parser.add_argument('--frac_train_to_remove', type=float, default=100, help='fraction of train data to remove.')
    parser.add_argument('--n_checkpoints', type=int, default=10, help='no. checkpoints to perform retraining.')

    # Additional settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
