"""
Experiment: Generates an instance-attribution explanation for a test instance, sorts training
instances by influence, then removes and retrains a new tree ensemble on
this new dataset. It then re-predicts on the test instance and measures the change in
log loss. If these intances are important, than the log loss should increase.
"""
import argparse
from copy import deepcopy
import os
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

import trex
from utility import model_util, data_util, exp_util, print_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from maple.MAPLE import MAPLE


def measure_loglosses(sort_indices, x_test, x_test_label, X_train, y_train, clf):
    """
    Measures the change in log loss as training instances are removed.
    """
    new_model = clone(clf).fit(X_train, y_train)
    x_test_proba = new_model.predict_proba(x_test)
    x_test_logloss = log_loss([x_test_label], x_test_proba, labels=[0, 1])

    log_losses = [x_test_logloss]

    for percentage in tqdm.tqdm(range(10, 100, 10)):
        n_samples = int(X_train.shape[0] * (percentage / 100))
        remove_indices = sort_indices[:n_samples]
        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        # remeasure test instance log loss
        new_model = clone(clf).fit(new_X_train, new_y_train)
        x_test_proba = new_model.predict_proba(x_test)
        x_test_logloss = log_loss([x_test_label], x_test_proba, labels=[0, 1])
        log_losses.append(x_test_logloss)

    return log_losses


def evaluation(args):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # make logger
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    # get model and data
    clf = model_util.get_classifier(args.model, n_estimators=args.n_estimators, max_depth=args.max_depth,
                                    random_state=args.rs)
    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=args.rs,
                                                                 data_dir=args.data_dir)

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('test instances: {}\n'.format(len(X_test)))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)

    trex_res = []
    random_res = []
    pcts = list(range(0, 100, 10))

    seed = args.rs
    for i in range(args.repeats):
        seed += 1

        # pick a test instance to explain
        test_preds = model.predict(X_test)
        if args.misclassified:
            indices = np.where(y_test != test_preds)[0]
        else:
            indices = np.where(y_test == test_preds)[0]

        np.random.seed(seed)
        test_ndx = np.random.choice(indices)
        x_test = X_test[[test_ndx]]
        x_test_label = y_test[test_ndx]

        logger.info('\nRun {}, seed: {}, test index: {}'.format(i, seed, test_ndx))

        # our method
        logger.info('ordering by our method...')
        explainer = trex.TreeExplainer(model, X_train, y_train, encoding=args.encoding, dense_output=True,
                                       random_state=seed, use_predicted_labels=not args.true_label,
                                       kernel=args.kernel, linear_model=args.linear_model, C=args.C)
        trex_contribs = explainer.explain(x_test)[0]
        trex_contribs_sort_ndx = np.argsort(trex_contribs)[::-1]
        trex_pos_contribs = np.where(trex_contribs > 0)[0]

        trex_losses = measure_loglosses(trex_contribs_sort_ndx, x_test, x_test_label, X_train, y_train, clf)
        trex_res.append(trex_losses)

        # random method
        logger.info('ordering by random...')
        np.random.seed(seed)
        random_sort_ndx = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)
        random_losses = measure_loglosses(random_sort_ndx, x_test, x_test_label, X_train, y_train, clf)
        random_res.append(random_losses)

    # organize results
    trex_res = np.vstack(trex_res)
    random_res = np.vstack(random_res)

    trex_v_random = trex_res - random_res
    trex_v_random_mean, trex_v_random_std = np.mean(trex_v_random, axis=0), np.std(trex_v_random, axis=0)

    # plot results
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    ax = axs[0]
    ax.plot(pcts, trex_res[0], color='cyan', label='TREX-{}'.format(args.linear_model))
    ax.plot(pcts, random_res[0], color='red', label='Random')
    ax.set_xlabel('% train data removed')
    ax.set_ylabel('log loss')
    ax.legend()

    ax = axs[1]
    ax.errorbar(pcts, trex_v_random_mean, yerr=trex_v_random_std, color='green',
                label='TREX-{} - Random'.format(args.linear_model))
    ax.set_xlabel('% train data removed')
    ax.set_ylabel('log loss')
    ax.legend()

    plt.savefig(os.path.join(out_dir, '{}.pdf'.format(args.dataset)), bbox_inches='tight')

    # if args.save_results:
    #     if args.knn:
    #         np.save(os.path.join(efficiency_dir, 'knn_check_pct.npy'), knn_check_pct)
    #         np.save(os.path.join(efficiency_dir, 'knn_fix_pct.npy'), knn_fix_pct)
    #         np.save(os.path.join(effectiveness_dir, 'knn_check_pct.npy'), knn_check_pct)
    #         np.save(os.path.join(effectiveness_dir, 'knn_acc.npy'), knn_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/evaluation', help='directory to save results.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='tree model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')
    parser.add_argument('--kernel', default='linear', help='Similarity kernel for the linear model.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--linear_model_loss', action='store_true', default=False, help='Include linear loss.')
    parser.add_argument('--save_plot', action='store_true', default=False, help='Save plot results.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--flip_frac', type=float, default=0.4, help='Fraction of train labels to flip.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--maple', action='store_true', help='Whether to use MAPLE as a baseline.')
    parser.add_argument('--misclassified', action='store_true', help='explain misclassified test instance.')
    parser.add_argument('--repeats', type=int, default=5, help='Number of times to repeat the experiment.')
    parser.add_argument('--check_pct', type=float, default=0.3, help='Max percentage of train instances to check.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--gridsearch', action='store_true', default=False, help='Use gridsearch to tune KNN.')
    parser.add_argument('--knn_neighbors', type=int, default=5, help='Use KNN on top of TREX features.')
    parser.add_argument('--knn_weights', type=str, default='uniform', help='Use KNN on top of TREX features.')
    args = parser.parse_args()
    evaluation(args)
