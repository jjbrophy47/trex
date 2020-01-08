"""
Modular version of roar2, saves contributions of training points for each test point.
"""
import time
import argparse
from copy import deepcopy
import os
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import tqdm
import numpy as np
from sklearn.base import clone

import trex
from utility import model_util, data_util, print_util, exp_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from maple.MAPLE import MAPLE


def _our_method(X_test, tree, args, X_train, y_train, X_val, seed, logger, model_dir):

    # load previously saved model
    # model_path = os.path.join(model_dir, 'ours_{}.pkl'.format(args.linear_model))

    # if os.path.exists(model_path):
    #     logger.info('loading model from: {}'.format(model_path))
    #     explainer = trex.TreeExplainer.load(model_path)

    # else:
    explainer = trex.TreeExplainer(tree, X_train, y_train, encoding=args.encoding,
                                   dense_output=True, logger=logger, X_val=X_val,
                                   random_state=seed, use_predicted_labels=not args.true_label,
                                   kernel=args.kernel, linear_model=args.linear_model,
                                   verbose=args.verbose)
        # logger.info('saving model to: {}'.format(model_path))
        # explainer.save(model_path)

    # sort instanes with highest positive influence first
    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = explainer.explain(X_test[i].reshape(1, -1))[0]
        if args.linear_model == 'svm':
            n_sv = len(np.where(contributions != 0)[0])
            n_pos = len(np.where(contributions > 0)[0])
            sv_pct = (n_sv / X_train.shape[0]) * 100
            logger.info('support vectors: {} ({:.2f}%), positive sv: {}'.format(n_sv, sv_pct, n_pos))

        if args.trex_absolute:
            contributions = np.abs(contributions)

        contributions_sum += contributions
    return contributions_sum


def _maple_method(X_test, args, model, X_train, y_train, logger, model_dir):

    # load previously saved model
    # model_path = os.path.join(model_dir, 'maple.pkl')

    # if os.path.exists(model_path):
    #     logger.info('loading model from: {}'.format(model_path))
    #     explainer = MAPLE.load(model_path)

    # else:
    train_label = y_train if args.true_label else model.predict(X_train)
    explainer = MAPLE(X_train, train_label, X_train, train_label, verbose=args.verbose, dstump=False)
    logger.info('saving model to: {}'.format(model_path))
    explainer.save(model_path)

    # order the training instances
    train_weights = []
    for i in tqdm.tqdm(range(X_test.shape[0])):
        train_weights.append(explainer.get_weights(X_test[i]))
    train_weight = np.sum(np.vstack(train_weights), axis=0)
    return train_weight


def _influence_method(X_test, args, model, X_train, y_train, y_test, logger):

    model_path = '.model.json'
    model.save_model(model_path, format='json')

    if args.inf_k == -1:
        update_set = 'AllPoints'
    elif args.inf_k == 0:
        update_set = 'SinglePoint'
    else:
        update_set = 'TopKLeaves'

    explainer = CBLeafInfluenceEnsemble(model_path, X_train, y_train, k=args.inf_k,
                                        learning_rate=model.learning_rate_, update_set=update_set)

    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = []
        buf = deepcopy(explainer)

        for j in tqdm.tqdm(range(len(X_train))):
            explainer.fit(removed_point_idx=j, destination_model=buf)
            contributions.append(buf.loss_derivative(X_test[[i]], y_test[[i]])[0])

        contributions = np.array(contributions)
        contributions_sum += contributions

    return contributions_sum


def _knn_method(X_test, args, model, X_train, y_train, y_test, logger):

    # transform the data
    extractor = trex.TreeExtractor(model, encoding=args.encoding)
    X_train_alt = extractor.fit_transform(X_train)
    X_test_alt = extractor.transform(X_test)

    # setup aggregate data container
    contributions_sum = np.zeros(X_train.shape[0])
    train_label = y_train if args.true_label else model.predict(X_train)

    # compute the contribution of all training samples for each test instance
    for i in tqdm.tqdm(range(X_test.shape[0])):
        distances = np.linalg.norm(X_test_alt[i] - X_train_alt, axis=1)
        contributions = np.divide(1, distances, out=np.zeros_like(distances), where=distances != 0)

        if not args.knn_absolute:
            neg_ndx = np.where(train_label != y_test[i])[0]
            contributions[neg_ndx] *= -1

        contributions_sum += contributions

    return contributions_sum


def roar(args, logger, out_dir, seed):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # crete a models directory
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # get model and data
    clf = model_util.get_classifier(args.model, n_estimators=args.n_estimators, max_depth=args.max_depth,
                                    random_state=args.rs)
    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=1,
                                                                 data_dir=args.data_dir)

    # use part of the test data as validation data
    X_val = X_test.copy()
    if args.val_frac < 1.0 and args.val_frac > 0.0:
        X_val = X_val[int(X_val.shape[0] * args.val_frac):]

    # use part of the test data for evaluation
    if args.test_frac < 1.0 and args.test_frac > 0.0:
        n_test_samples = int(X_test.shape[0] * args.test_frac)
        X_test, y_test = X_test[:n_test_samples], y_test[:n_test_samples]

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('test instances: {}\n'.format(len(X_test)))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)

    # save model
    model_path = os.path.join(model_dir, 'tree.pkl')
    logger.info('saving model to: {}'.format(model_path))
    exp_util.save_model(model, model_path)

    # pick a test instance to explain
    np.random.seed(seed)
    test_ndx = np.random.choice(np.arange(X_test.shape[0]))
    X_test = X_test[[test_ndx]]
    y_test = y_test[[test_ndx]]
    logger.info('test instance: {}'.format(test_ndx))

    # our method
    if args.trex:
        logger.info('computing contributions by our method...')
        start = time.time()
        our_res = _our_method(X_test, model, args, X_train, y_train, X_val, seed, logger, model_dir)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # MAPLE method
    if args.maple:
        logger.info('computing contributions by MAPLE...')
        start = time.time()
        maple_res = _maple_method(X_test, args, model, X_train, y_train, logger, model_dir)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # influence method
    if args.model == 'cb' and args.inf_k is not None:
        logger.info('computing contributions by LeafInfluence...')
        start = time.time()
        leafinfluence_res = _influence_method(X_test, args, model, X_train, y_train, y_test, logger)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # KNN method
    if args.knn:
        logger.info('computing contributions by knn...')
        start = time.time()
        knn_res = _knn_method(X_test, args, model, X_train, y_train, y_test, logger)
        logger.info('time: {:3f}s'.format(time.time() - start))

    if args.save_results:

        # make seed directory
        rs_dir = os.path.join(out_dir, 'rs{}'.format(seed))
        os.makedirs(rs_dir, exist_ok=True)

        # trex
        if args.trex:
            np.save(os.path.join(rs_dir, 'ours_{}.npy'.format(args.linear_model)), our_res)

        # MAPLE
        if args.maple:
            np.save(os.path.join(rs_dir, 'maple.npy'), maple_res)

        # TEKNN
        if args.knn:
            np.save(os.path.join(rs_dir, 'teknn.npy'), knn_res)

        # LeafInfluence
        if args.model == 'cb' and args.inf_k is not None:
            np.save(os.path.join(rs_dir, 'leafinfluence.npy'), leafinfluence_res)


def main(args):

    # make logger
    dataset = args.dataset
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        dataset += '_{}'.format(str(args.train_frac).replace('.', 'p'))
    out_dir = os.path.join(args.out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    # run experiment
    seed = args.rs
    for i in range(args.repeats):
        logger.info('\nRun {}, seed: {}'.format(i + 1, seed))
        roar(args, logger, out_dir, seed=seed)
        seed += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/roar3', help='directory to save results.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')
    parser.add_argument('--test_frac', type=float, default=1.0, help='dataset to evaluate on.')
    parser.add_argument('--model', type=str, default='cb', help='tree model to use.')
    parser.add_argument('--trex', action='store_true', help='Use TREX.')
    parser.add_argument('--trex_absolute', action='store_true', help='Absolute values of TREX contributions.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')
    parser.add_argument('--kernel', default='linear', help='Similarity kernel for the linear model.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=1, help='for reproducibility.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--maple', action='store_true', help='Whether to use MAPLE as a baseline.')
    parser.add_argument('--misclassified', action='store_true', help='explain misclassified test instance.')
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat the experiment.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--knn_absolute', action='store_true', help='Absolute values of KNN contributions.')
    args = parser.parse_args()
    main(args)
