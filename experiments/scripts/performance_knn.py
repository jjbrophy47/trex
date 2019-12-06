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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from utility import model_util, data_util, print_util


def performance(args, model_type='cb', encoding='leaf_output', dataset='adult', n_estimators=100, random_state=69,
                gridsearch=False, verbose=0, data_dir='data', out_dir='output/performance_knn/'):
    """
    Main method comparing performance of tree ensembles and svm models.
    """

    # write output to logs
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}_{}.txt'.format(dataset, model_type)))
    logger.info(args)

    # get model and data
    clf = model_util.get_classifier(model_type, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('num features: {}'.format(X_train.shape[1]))

    # train a tree ensemble
    logger.info('\ntree ensemble')
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)

    # train a knn model
    logger.info('\nknn')
    clf = KNeighborsClassifier()
    if gridsearch:
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 31, 45, 61], 'weights': ['uniform', 'distance']}
        gs = GridSearchCV(clf, param_grid, cv=5, verbose=verbose).fit(X_train, y_train)
        knn = gs.best_estimator_
        logger.info(gs.best_params_)
    else:
        knn = KNeighborsClassifier().fit(X_train, y_train)
    model_util.performance(knn, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--gridsearch', action='store_true', default=False, help='gridsearch for SVM model.')
    parser.add_argument('--verbose', metavar='LEVEL', default=0, type=int, help='verbosity of gridsearch output.')
    args = parser.parse_args()
    performance(args, args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.gridsearch,
                args.verbose)
