"""
Extracts different feature representations of instances from a tree ensemble.
Currently supports: LightGBM, RandomForestClassifier (sklearn).
In the future: CatBoost, XGBoost.
"""
import time
import argparse
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder


def tree_path_encoding(model, X, one_hot_enc=None, to_dense=True):
    """
    Encodes each x in X as a binary vector whose length is equal to the number of
    leaves or nodes in the ensemble, with 1's representing the instance ending at that leaf,
    0 otherwise.
    """
    assert X.ndim == 2, 'X is not 2d!'
    start = time.time()

    # get the leaf ids and num leaves or nodes of each tree for all instances
    if str(model).startswith('RandomForestClassifier'):
        leaves = model.apply(X)
        leaves_per_tree = _parse_rf_model(model, nodes_per_tree=True)  # actually using nodes, could refine

    elif str(model).startswith('LGBMClassifier'):
        leaves = model.predict_proba(X, pred_leaf=True)
        leaves_per_tree = _parse_lgb_model(model, leaves_per_tree=True)

    if one_hot_enc is None:

        # make sure all leaves have been seen at least once
        assert np.all(np.max(leaves, axis=0) + 1 == leaves_per_tree), 'lgb leaves do not match max leaves found'
        one_hot_enc = OneHotEncoder(categories='auto').fit(leaves)

    encoding = one_hot_enc.transform(leaves)
    if to_dense:
        encoding = np.array(encoding.todense())

    print('path encoding time: {:.3f}'.format(time.time() - start))

    return encoding, one_hot_enc


def tree_output_encoding(model, X):
    """
    Encodes each x in X as a concatenation of one-hot encodings, one for each tree.
    Each one-hot encoding represents the class or output at the leaf x traversed to.
    All one-hot encodings are concatenated, to get a vector of size n_trees * n_classes.
    """
    assert X.ndim == 2, 'X is not 2d!'
    start = time.time()

    if str(model).startswith('RandomForestClassifier'):
        one_hot_preds = [tree.predict_proba(X) for tree in model.estimators_]
        encoding = np.hstack(one_hot_preds)

    # DO NOT USE! An instance can be more similar with other instances than itself, since these are leaf values
    elif str(model).startswith('LGBMClassifier'):
        exit('tree output encoding for LGB not good, do not use!')
        leaves = model.predict_proba(X, pred_leaf=True)
        encoding = np.zeros(leaves.shape)

        for i in range(leaves.shape[0]):  # per instance
            for j in range(leaves.shape[1]):  # per tree
                encoding[i][j] = model.booster_.get_leaf_output(j, leaves[i][j])

    else:
        exit('model {} not implemented!'.format(str(model)))

    print('output encoding time: {:.3f}'.format(time.time() - start))
    return encoding


def _parse_rf_model(model, leaves_per_tree=False, nodes_per_tree=False):
    """Returns low-level information about sklearn's RandomForestClassifier."""

    result = None

    if leaves_per_tree:
        result = np.array([tree.get_n_leaves() for tree in model.estimators_])

    elif nodes_per_tree:
        result = np.array([tree.tree_.node_count for tree in model.estimators_])

    return result


def _parse_lgb_model(model, leaves_per_tree=False):
    """Returns the low-level data structure of the lgb model."""

    model_dict = model.booster_.dump_model()

    result = model_dict

    if leaves_per_tree:
        result = np.array([tree_dict['num_leaves'] for tree_dict in model_dict['tree_info']])

    return result


def _similarity(x_feature, X_feature, X_train=None, k=5):
    """Finds which instances are most similar to x_feature."""

    if x_feature.ndim == 2:
        x_feature = x_feature[0]

    sim = np.matmul(X_feature, x_feature)
    sim_ndx = np.argsort(sim)[::-1][:k]

    # display most similar train instances
    if X_train is not None:
        print('\nSimilar Train Instances')
        for ndx in sim_ndx:
            print('\nTrain [{}], similarity: {:.3f}'.format(ndx, sim[ndx]))
            print(X_train[ndx])

    return sim, sim_ndx


def _euclidean(x, X, k=5):
    """Computes the euclidean distance of x to each instance in X."""

    dist = np.linalg.norm(x - X, axis=1)
    dist_ndx = np.argsort(dist)[:k]
    return dist, dist_ndx


def _plot_sim_dist(ndx, sim, dist, sim_ndx):
    """Plots similarity vs euclidean distance, ordered by most similar train instances."""

    sim_vals = sim[sim_ndx]
    dist_vals = dist[sim_ndx]
    pearson = np.corrcoef(sim_vals, dist_vals)[0][1]

    fig, ax = plt.subplots()
    ax.scatter(dist_vals, sim_vals)
    ax.set_title('Test [{}], Similarity vs Euclidean, correlation: {:.3f}'.format(ndx, pearson))
    ax.set_xlabel('distance')
    ax.set_ylabel('similarity')
    plt.show()


def _svm_prediction(svm, x_feature, train_feature, y_train):
    """
    Computes the prediction for a query point as a weighted sum of support vectors.
    This should match the `svm.decision_function` method.
    """
    assert x_feature.ndim == 2, 'x_feature is not 2d!'

    sv_feature = train_feature[svm.support_]  # support vector train instances
    sv_weight = svm.dual_coef_[0]  # support vector weights

    sim_prod = np.matmul(sv_feature, x_feature[0])
    weighted_prod = sim_prod * sv_weight
    prediction = (np.sum(weighted_prod) + svm.intercept_)[0]

    return prediction, weighted_prod


def main(args):
    clf = None

    # create model
    if args.model == 'lgb':
        clf = lgb.LGBMClassifier(random_state=args.rs, n_estimators=args.n_estimators)
    elif args.model == 'rf':
        clf = RandomForestClassifier(random_state=args.rs, n_estimators=args.n_estimators)

    # load dataset
    if args.dataset == 'iris':
        data = load_iris()
    elif args.dataset == 'breast':
        data = load_breast_cancer()
    elif args.dataset == 'wine':
        data = load_wine()

    X = data['data']
    y = data['target']
    label = data['target_names']

    print('label names: {}'.format(label))

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.rs, stratify=y)

    # train model
    model = clone(clf).fit(X_train, y_train)
    y_hat = model.predict(X_test)
    print('test set acc: {:4f}'.format(accuracy_score(y_test, y_hat)))

    # extract feature representations
    if args.encoding == 'path':
        train_feature, train_enc = tree_path_encoding(model, X_train)
        test_feature, _ = tree_path_encoding(model, X_test, one_hot_enc=train_enc)

    elif args.encoding == 'output':
        exit('tree output encodings under construction...')
        train_feature = tree_output_encoding(model, X_train)
        test_feature = tree_output_encoding(model, X_test)

    else:
        exit('encoding {} not implemented!'.format(args.encoding))

    # test train or test instance similarity to train instances
    ndx = 2
    x = X_test[ndx]
    x_feature = test_feature[ndx].reshape(1, -1)

    print('\nTest [{}]'.format(ndx))
    print(X_test[ndx])

    sim, sim_ndx = _similarity(x_feature, train_feature, X_train=X_train, k=args.topk)
    dist, _ = _euclidean(x, X_train)
    # _plot_sim_dist(ndx, sim, dist, sim_ndx)

    # train an SVM on the feature representations
    svm = SVC(kernel=lambda x, y: np.dot(x, y.T), random_state=args.rs).fit(train_feature, y_train)
    y_hat_svm = svm.predict(test_feature)
    print('\ntest set acc (svm): {:4f}'.format(accuracy_score(y_test, y_hat_svm)))

    num_same_preds = np.count_nonzero(np.where(y_hat == y_hat_svm))
    print('svm same predictions: {} / {}'.format(num_same_preds, len(y_hat)))

    print('support vectors (train indicies):')
    print(svm.support_)

    # explain a test instance prediction
    prediction, influence = _svm_prediction(svm, x_feature, train_feature, y_train)
    prediction_label = svm.predict(x_feature)
    decision_pred = svm.decision_function(x_feature)[0]
    assert np.isclose(prediction, decision_pred), 'svm.decision_function does not match _svm_prediction!'

    # sort most influential train instances
    neg_inf_ndx = np.where(influence < 0)[0]
    neg_inf = influence[neg_inf_ndx]
    neg_inf_sv_ndx = svm.support_[neg_inf_ndx]
    neg_inf_list = sorted(zip(neg_inf_sv_ndx, neg_inf), key=lambda tup: tup[1])

    pos_inf_ndx = np.where(influence > 0)[0]
    pos_inf = influence[pos_inf_ndx]
    pos_inf_sv_ndx = svm.support_[pos_inf_ndx]
    pos_inf_list = sorted(zip(pos_inf_sv_ndx, pos_inf), key=lambda tup: tup[1], reverse=True)

    # show most influential train instances
    # TODO: add label
    print('\nTest [{}], distance to separator: {:.3f}, prediction: {}'.format(ndx, prediction, prediction_label))
    print(X_test[ndx])

    print('\nExcitatory Train Instances')
    for ndx, inf in pos_inf_list[:args.topk]:
        # TODO: add similarity and label
        print('\nTrain [{}], impact: {:.3f}'.format(ndx, inf))
        print(X_train[ndx])

    print('\nInhibitory Train Instances')
    for ndx, inf in neg_inf_list[:args.topk]:
        print('\nTrain [{}], impact: {:.3f}'.format(ndx, inf))
        print(X_train[ndx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk', metavar='NUM', type=int, default=5, help='Num of similar instances to display.')
    args = parser.parse_args()
    print(args)

    main(args)
