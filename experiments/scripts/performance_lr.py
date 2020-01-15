"""
Experiment: Tests tree ensemble v SVM v SVM trained on tree ensemble feature representations for performance.
If an SVM is already as good as a tree ensemble, there is no need to explain a tree ensemble with an SVM.
"""
import os
import sys
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner; TODO: remove this dependency

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from utility import model_util, data_util, print_util


def performance(args):
    """
    Main method comparing performance of tree ensembles and svm models.
    """

    # write output to logs
    os.makedirs(args.out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(args.out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    # get model and data
    clf = model_util.get_classifier(args.model, n_estimators=args.n_estimators, random_state=args.rs)
    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=args.rs,
                                                                 data_dir=args.data_dir)

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('num features: {}'.format(X_train.shape[1]))

    # train a tree ensemble
    logger.info('\ntree ensemble')
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)

    # train an svm
    logger.info('\nlr')
    clf = LogisticRegression()
    if args.gs:
        param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]}
        gs = GridSearchCV(clf, param_grid, cv=2, verbose=args.verbose).fit(X_train, y_train)
        linear_model = gs.best_estimator_
        logger.info(gs.best_params_)
    else:
        linear_model = LogisticRegression(penalty=args.penalty, C=args.C).fit(X_train, y_train)
    model_util.performance(linear_model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/performance_lr', help='output directory.')
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--penalty', type=str, default='l1', help='LR kernel.')
    parser.add_argument('--C', type=float, default=0.1, help='LR penalty.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=1, help='for reproducibility.')
    parser.add_argument('--gs', action='store_true', default=False, help='gridsearch for SVM model.')
    parser.add_argument('--cv', type=int, default=2, help='number of cross-validation folds.')
    parser.add_argument('--verbose', metavar='LEVEL', default=0, type=int, help='verbosity of gridsearch output.')
    args = parser.parse_args()
    performance(args)
