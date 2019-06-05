"""
Idenitfies the impact of training instances on a test instance for a tree-ensemble classifier.
"""
import shap
import tqdm
import time
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.base import clone
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from . import waterfall

from importlib import reload
reload(waterfall)


def _print_feature_value(feature_name, feature_value):
    for feature, value in zip(feature_name, feature_value):
        print('\t{}: {:.2f}'.format(feature, value))


def _print_instance(ndx, feature_name, x, actual, predicted=None, impact=None):
    if predicted is not None:
        s1 = '\nTEST INSTANCE [{}], predicted: {}, label: {}'.format(ndx, predicted, actual)
    else:
        assert impact is not None
        s1 = '\nTRAIN INSTANCE [{}], label: {}, impact: {:.2f}'.format(ndx, actual, impact)

    print(s1)


def instance_overlap(inf_a, inf_b, k=5):

    print('\nTrain instance overlap:')

    infa_ndx = np.argsort(inf_a)[::-1]
    infb_ndx = np.argsort(inf_b)[::-1]

    posa_ndx = [x for x in infa_ndx[:k] if inf_a[x] > 0]
    posb_ndx = [x for x in infb_ndx[:k] if inf_b[x] > 0]

    nega_ndx = [x for x in infa_ndx[::-1][:k] if inf_a[x] < 0]
    negb_ndx = [x for x in infb_ndx[::-1][:k] if inf_b[x] < 0]

    posa_posb = set(posa_ndx).intersection(posb_ndx)
    posa_negb = set(posa_ndx).intersection(negb_ndx)

    nega_posb = set(nega_ndx).intersection(posb_ndx)
    nega_negb = set(nega_ndx).intersection(negb_ndx)

    print('pos (sim) - pos (ret): {}'.format(posa_posb))
    print('neg (sim) - pos (ret): {}'.format(nega_posb))

    print('\npos (sim) - neg (ret): {}'.format(posa_negb))
    print('neg (sim) - neg (ret): {}'.format(nega_negb))


def remove_sample(arr, ndx, output='index'):
    """Reindexes an array with the specified `ndx` removed."""

    new_index = np.arange(len(arr))
    new_index = new_index[np.where(new_index != ndx)]

    result = new_index
    if output == 'array':
        result = arr[new_index]
    elif output == 'both':
        result = arr[new_index], new_index
    return result


def display_topk(inf, X_train, feature, y_train, label, k=5, plot_waterfall=False, shap_data=None):
    inf_ndx = np.argsort(inf)[::-1]

    pos_ndx = inf_ndx[:k]
    neg_ndx = inf_ndx[::-1][:k]

    print('\n\nEXCITATORY TRAIN INSTANCES'.format(k))
    for ndx in pos_ndx:
        if inf[ndx] > 0:
            _print_instance(ndx, feature, X_train[ndx], label[y_train[ndx]], impact=inf[ndx])

            if shap_data is not None:
                exp_val, shap_test, shap_train = shap_data

                if plot_waterfall:
                    waterfall.waterfall(shap_train[ndx], feature, feature_vals=X_train[ndx], base_level=exp_val)
                else:
                    shap.force_plot(exp_val, shap_train[ndx], X_train[ndx], feature_names=feature, matplotlib=True)

    print('\n\nINHIBITORY TRAIN INSTANCES'.format(k))
    for ndx in neg_ndx:
        if inf[ndx] < 0:
            _print_instance(ndx, feature, X_train[ndx], label[y_train[ndx]], impact=inf[ndx])

            if shap_data is not None:
                exp_val, shap_test, shap_train = shap_data

                if plot_waterfall:
                    waterfall.waterfall(shap_train[ndx], feature, feature_vals=X_train[ndx], base_level=exp_val)
                else:
                    shap.force_plot(exp_val, shap_train[ndx], X_train[ndx], feature_names=feature, matplotlib=True)


def train_impact_retrain(model, x_test, X_train, Y_train):
    """Measures train instance impact by re-training without each training instance."""

    y_hat = model.predict_proba(x_test)
    y_hat_label = np.argmax(y_hat)

    influence = []
    for i in tqdm.tqdm(range(len(X_train))):
        new_index = remove_sample(X_train, i)
        new_X_train = X_train[new_index]
        new_Y_train = Y_train[new_index]

        model = clone(model).fit(new_X_train, new_Y_train)
        new_y_hat = model.predict_proba(x_test)

        impact = y_hat - new_y_hat
        influence.append(impact[0][y_hat_label])

    return influence


def train_impact_heuristic(model, x_test, X_train, Y_train):
    """Measures train instance impact to test instance."""

    # get predicted label for the test instance
    y_hat_probs = model.predict_proba(x_test)
    y_hat_label = np.argmax(y_hat_probs)

    # get the leaf ids for each tree for train and test instances
    if str(model).startswith('RandomForestClassifier'):
        x_test_leaves = model.apply(x_test)
        X_train_leaves = model.apply(X_train)
    if str(model).startswith('LGBMClassifier'):
        x_test_leaves = model.predict_proba(x_test, pred_leaf=True)[0]
        X_train_leaves = model.predict_proba(X_train, pred_leaf=True)

    # quantify the impact of each training instance on the test instance
    influence = []
    for x_train_leaves, y_train in zip(X_train_leaves, Y_train):

        # compute similarity between train and test instance by looking at decision path overlaps
        similarity = np.mean(x_train_leaves == x_test_leaves)

        # positive influence if the train label matches the predicted test label, otherwise negative influence
        if y_train != y_hat_label and similarity > 0:
            similarity *= -1

        influence.append(similarity)

    return influence


def explain_instance(model, test_ndx, x_test, X_test, y_test, X_train, y_train, label, feature,
                     k=5, method='both', shap_data=None, plot_waterfall=False):
    """Explain a test instance using the most impactful training instances."""

    # get predicted label for the test instance
    y_hat_probs = model.predict_proba(x_test)
    y_hat_label = np.argmax(y_hat_probs)

    x_test_contrib = None

    if str(model).startswith('LGBMClassifier'):
        n_class = len(y_hat_probs[0])
        n_feat = len(x_test[0])
        x_test_contrib = model.predict_proba(x_test, pred_contrib=True)

        if n_class == 2:
            x_test_contrib = x_test_contrib[0]
        else:
            assert n_class > 2
            x_test_contrib = x_test_contrib.reshape(n_class, n_feat + 1)[y_hat_label]

        feature_contrib = x_test_contrib[:n_feat]
        base_level = x_test_contrib[-1]

    # display instance
    _print_instance(test_ndx, feature, x_test[0], label[y_test[test_ndx]], predicted=label[y_hat_label])

    # display feature contributions
    if plot_waterfall:
        waterfall.waterfall(feature_contrib, feature, feature_vals=x_test[0], base_level=base_level)
    else:
        if shap_data is not None:
            exp_val, shap_test, shap_train = shap_data
            shap.force_plot(exp_val, shap_test[test_ndx], X_test[test_ndx], feature_names=feature, matplotlib=True)

    if method in ('similarity', 'both'):
        start = time.time()
        inf_a = train_impact_heuristic(model, x_test, X_train, y_train)
        end = time.time() - start
        print('\nsimilarity took {:4f} seconds'.format(end))
        print('\n\nSIMILARTIY')
        display_topk(inf_a, X_train, feature, y_train, label, k=k, shap_data=shap_data, plot_waterfall=plot_waterfall)

    if method in ('retrain', 'both'):
        start = time.time()
        inf_b = train_impact_retrain(model, x_test, X_train, y_train)
        end = time.time() - start
        print('retraining took {:4f} seconds'.format(end))
        print('\n\nRETRAINING')
        display_topk(inf_b, X_train, feature, y_train, label, k=k, shap_data=shap_data, plot_waterfall=plot_waterfall)

    if method == 'both':
        instance_overlap(inf_a, inf_b, k=k)


def main(args):
    clf = lgb.LGBMClassifier(random_state=args.rs, n_estimators=args.n_estimators)

    if args.dataset == 'iris':
        data = load_iris()
    elif args.dataset == 'breast':
        data = load_breast_cancer()
    elif args.dataset == 'wine':
        data = load_wine()

    X = data['data']
    y = data['target']
    label = data['target_names']
    feature = data['feature_names']

    print('label names: {}'.format(label))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.rs, stratify=y)

    model = clone(clf).fit(X_train, y_train)
    y_hat = model.predict(X_test)
    print('test set acc: {:4f}'.format(accuracy_score(y_test, y_hat)))

    print('feature importance:')
    _print_feature_value(feature, model.feature_importances_)

    # mispredicted test instances
    y_miss = np.where(y_test != y_hat)[0]
    print('misspredicted test instances: {}'.format(y_miss))

    for miss_ndx in y_miss:
        x_test = X_test[miss_ndx].reshape(1, -1)
        explain_instance(model, miss_ndx, x_test, y_test, y_hat, X_train, y_train, label, feature, k=args.k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train instance explanations for random forest',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--topk', metavar='N', type=int, default=5, help='top n train instances to display.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    args = parser.parse_args()

    main(args)
