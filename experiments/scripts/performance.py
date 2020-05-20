"""
Model performance.
"""
import os
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from utility import model_util
from utility import data_util
from utility import print_util


def _get_classifier(args):
    """
    Return the appropriate classifier.
    """
    if args.model in ['cb', 'lgb', 'xgb']:
        clf = model_util.get_classifier(args.model,
                                        n_estimators=args.n_estimators,
                                        max_depth=args.max_depth,
                                        random_state=args.rs)
        params = {'n_estimators': [10, 100, 250], 'max_depth': [1, 3, 5, 10, None]}

    elif args.model == 'lr':
        clf = LogisticRegression(penalty=args.penalty, C=args.C)
        params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]}

    elif args.model == 'svm_linear':
        clf = LinearSVC(dual=False, penalty=args.penalty, C=args.C)
        params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]}

    elif args.model == 'svm_rbf':
        clf = SVC(gamma='auto', penalty=args.penalty, C=args.C, kernel=args.kernel)
        params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]}

    elif args.model == 'knn':
        clf = KNeighborsClassifier(weights=args.weights, n_neighbors=args.n_neighbors)
        params = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]}

    else:
        raise ValueError('model uknown: {}'.format(args.model))

    return clf, params


def experiment(args, logger, out_dir, seed):
    """
    Main method comparing performance of tree ensembles and svm models.
    """

    # get model and data
    clf, params = _get_classifier(args)
    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir)
    X_train, X_test, y_train, y_test, label = data

    logger.info('train instances: {:,}'.format(len(X_train)))
    logger.info('test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # train model
    logger.info('\nmodel: {}, params: {}'.format(args.model, params))

    if args.tune:
        gs = GridSearchCV(clf, params, cv=args.cv, verbose=args.verbose).fit(X_train, y_train)

        cols = ['mean_fit_time', 'mean_test_score', 'rank_test_score']
        cols += ['param_{}'.format(param) for param in params.keys()]

        df = pd.DataFrame(gs.cv_results_)
        logger.info('gridsearch results:')
        logger.info(df[cols].sort_values('rank_test_score'))

        model = gs.best_estimator_
        logger.info('best params: {}'.format(gs.best_params_))

    else:
        model = clf.fit(X_train, y_train)

    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, args.model)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/performance/', help='output directory.')

    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--tune', action='store_true', default=True, help='whether to tune the model.')
    parser.add_argument('--cv', type=int, default=3, help='number of cross-val folds.')

    # tree hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth in tree ensemble.')

    # LR and SVM hyperparameters
    parser.add_argument('--penalty', type=str, default='l2', help='linear model penalty type.')
    parser.add_argument('--C', type=float, default=0.1, help='linear model penalty.')
    parser.add_argument('--kernel', type=str, default='rbf', help='linear model kernel.')

    # knn hyperparameters
    parser.add_argument('--weights', type=str, default='uniform', help='knn weights.')
    parser.add_argument('--n_neighbors', type=int, default=3, help='number of k nearest neighbors.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', default=2, type=int, help='verbosity level.')

    args = parser.parse_args()
    main(args)
