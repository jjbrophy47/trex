"""
Model performance.
"""
import os
import sys
import time
import resource
import argparse
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import clone

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import util


def get_model(args, cat_indices=None):
    """
    Return the appropriate classifier.
    """
    if args.model in ['cb', 'lgb', 'xgb']:
        clf = util.get_model(args.model,
                             n_estimators=args.n_estimators,
                             max_depth=args.max_depth,
                             random_state=args.rs,
                             cat_indices=cat_indices)
        params = {'n_estimators': [10, 25, 50, 100, 250], 'max_depth': [3, 5, 7]}

    elif args.model == 'rf':
        clf = util.get_model(args.model,
                             n_estimators=args.n_estimators,
                             max_depth=args.max_depth,
                             random_state=args.rs)
        params = {'n_estimators': [10, 25, 50, 100, 250], 'max_depth': [3, 5, 7],
                  'class_weight': [None, 'balanced']}

    elif args.model == 'dt':
        clf = DecisionTreeClassifier(random_state=args.rs)
        params = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                  'max_depth': [3, 5, 10, None]}

    elif args.model == 'lr':
        clf = Pipeline(steps=[
            ('ss', StandardScaler()),
            ('lr', LogisticRegression(penalty=args.penalty, C=args.C, solver='liblinear', random_state=args.rs))
        ])
        params = {'lr__penalty': ['l1', 'l2'], 'lr__C': [1e-2, 1e-1, 1e0]}

    elif args.model == 'svm_linear':
        clf = Pipeline(steps=[
            ('ss', StandardScaler()),
            ('svm', LinearSVC(dual=False, penalty=args.penalty, C=args.C, random_state=args.rs))
        ])
        params = {'svm__penalty': ['l1', 'l2'], 'svm__C': [1e-2, 1e-1, 1e0]}

    elif args.model == 'svm_rbf':
        clf = Pipeline(steps=[
            ('ss', StandardScaler()),
            ('svm', SVC(gamma='auto', C=args.C, kernel=args.kernel, random_state=args.rs))
        ])
        params = {'svm__C': [1e-2, 1e-1, 1e0]}

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

    # start experiment timer
    begin = time.time()

    # get data
    data = util.get_data(args.dataset,
                         data_dir=args.data_dir,
                         preprocessing=args.preprocessing)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    logger.info('no. train: {:,}'.format(X_train.shape[0]))
    logger.info('no. test: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # tune on a fraction of the training data
    if not args.no_tune:

        if args.tune_frac < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=2,
                                         train_size=args.tune_frac,
                                         random_state=args.rs)
            tune_indices, _ = list(sss.split(X_train, y_train))[0]
            X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]
            logger.info('tune instances: {:,}'.format(X_train_sub.shape[0]))

        else:
            X_train_sub, y_train_sub = X_train, y_train
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # get model
    model, param_grid = get_model(args, cat_indices=cat_indices)
    logger.info('\nmodel: {}, param_grid: {}'.format(args.model, param_grid))

    # tune the model
    start = time.time()
    if not args.no_tune:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.rs)
        gs = GridSearchCV(model, param_grid, scoring=args.scoring, cv=skf, verbose=args.verbose)
        gs = gs.fit(X_train_sub, y_train_sub)

        cols = ['mean_fit_time', 'mean_test_score', 'rank_test_score']
        cols += ['param_{}'.format(param) for param in param_grid.keys()]

        df = pd.DataFrame(gs.cv_results_)
        logger.info('gridsearch results:')
        logger.info(df[cols].sort_values('rank_test_score'))

        model = clone(gs.best_estimator_)
        logger.info('best params: {}'.format(gs.best_params_))

    tune_time = time.time() - start
    logger.info('tune time: {:.3f}s'.format(tune_time))

    # train model
    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info('train time: {:.3f}s'.format(train_time))

    # evaluate
    auc, acc, ap, ll = util.performance(model, X_test, y_test, logger, name=args.model)

    # save results
    result = {}
    result['model'] = args.model
    result['auc'] = auc
    result['acc'] = acc
    result['ap'] = ap
    result['ll'] = ll
    result['tune_time'] = tune_time
    result['train_time'] = train_time
    result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['tune_frac'] = args.tune_frac
    if args.model in ['cb', 'rf']:
        result['n_estimators'] = gs.best_params_['n_estimators']
        result['max_depth'] = gs.best_params_['max_depth']
    np.save(os.path.join(out_dir, 'results.npy'), result)

    # Macs show this in bytes, unix machines show this in KB
    logger.info('max_rss: {:,}'.format(result['max_rss']))
    logger.info('total time: {:.3f}s'.format(time.time() - begin))
    logger.info('saving results to {}...'.format(os.path.join(out_dir, 'results.npy')))


def main(args):

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.preprocessing,
                           'rs_{}'.format(args.rs))

    # create outut directory and clear any previous contents
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # write everything printed to stdout to this log file
    logfile, stdout, stderr = util.stdout_stderr_to_log(os.path.join(out_dir, 'log+.txt'))

    # run experiment
    experiment(args, logger, out_dir, seed=args.rs)

    # restore original stdout and stderr settings
    util.reset_stdout_stderr(logfile, stdout, stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/performance/', help='output directory.')

    # Experiment settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--no_tune', action='store_true', default=False, help='do not tune the model.')
    parser.add_argument('--cv', type=int, default=5, help='number of cross-val folds.')
    parser.add_argument('--scoring', type=str, default='accuracy', help='scoring metric.')
    parser.add_argument('--tune_frac', type=float, default=1.0, help='fraction of train data to use for tuning.')

    # tree hyperparameters
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth in tree ensemble.')

    # LR and SVM hyperparameters
    parser.add_argument('--penalty', type=str, default='l2', help='linear model penalty type.')
    parser.add_argument('--C', type=float, default=0.1, help='linear model penalty.')
    parser.add_argument('--kernel', type=str, default='rbf', help='linear model kernel.')

    # knn hyperparameters
    parser.add_argument('--weights', type=str, default='uniform', help='knn weights.')
    parser.add_argument('--n_neighbors', type=int, default=3, help='number of k nearest neighbors.')

    # Extra settings
    parser.add_argument('--verbose', default=2, type=int, help='verbosity level.')

    args = parser.parse_args()
    main(args)
