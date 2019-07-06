"""
Exploration: Examine the raw image data from the most impactful train instances for a given test instance.
    Overlay the binary or manipulation mask over the original probe image.
"""
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA

from util import model_util, data_util


def _sort_impact(sv_ndx, impact):
    """Sorts support vectors by absolute impact values."""

    if impact.ndim == 2:
        impact = np.sum(impact, axis=1)
    impact_list = zip(sv_ndx, impact)
    # impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)
    impact_list = sorted(sorted(impact_list, key=lambda x: x[0]), key=lambda x: abs(x[1]), reverse=True)

    sv_ndx, impact = zip(*impact_list)
    sv_ndx = np.array(sv_ndx)
    return sv_ndx, impact


def _display_image(x, ax=None, pred=None, actual=None, impact=None):

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(x.reshape(28, 28), cmap='gray')

    s = ''

    if pred is not None:
        s += 'predicted: {}\n'.format(pred)

    if actual is not None:
        s += 'label: {}'.format(actual)

    if impact is not None:
        s += '\nimpact: {:.3f}'.format(impact)

    ax.axis('off')
    ax.set_title(s, fontsize=11)


def prediction_explanation(model='lgb', encoding='tree_path', dataset='MFC18_EvalPart1', n_estimators=100,
                           random_state=69, topk_train=5, test_size=0.1, alpha=0.5, show_performance=True,
                           true_label=False, pca_components=50, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_image_id=True,
                              test_size=test_size)
    X_train, X_test, y_train, y_test, label = data

    print('train instances: {}'.format(len(X_train)))
    print('test instances: {}'.format(len(X_test)))
    print('labels: {}'.format(label))

    if pca_components is not None:
        print('reducing dimensionality from {} to {} using PCA...'.format(X_train.shape[1], pca_components))
        pca = PCA(pca_components, random_state=random_state).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

    # fit a tree ensemble and an explainer for that tree ensemble
    print('fitting {} model...'.format(model))
    tree = clone(clf).fit(X_train_pca, y_train)

    print('fitting tree explainer...')
    explainer = sexee.TreeExplainer(tree, X_train_pca, y_train, encoding=encoding, timeit=True,
                                    use_predicted_labels=not true_label, random_state=random_state)
    print(explainer)

    if show_performance:
        model_util.performance(tree, X_train_pca, y_train, X_test_pca, y_test)

    # pick a random manipulated test instance to explain
    np.random.seed(random_state)
    test_ndx = np.random.choice(y_test)
    x_test = X_test_pca[test_ndx]
    test_pred = int(tree.predict(x_test.reshape(1, -1))[0])
    test_actual = y_test[test_ndx]

    sv_ndx, impact = explainer.train_impact(x_test)
    sv_ndx, impact = _sort_impact(sv_ndx, impact)

    # print impactful train instances
    impact_list = list(zip(sv_ndx, impact))
    pos_impact_list = [impact_item for impact_item in impact_list if impact_item[1] > 0]
    neg_impact_list = [impact_item for impact_item in impact_list if impact_item[1] < 0]
    svm_pred, pred_label = explainer.decision_function(x_test, pred_svm=True)

    # show the test image
    fig, ax = plt.subplots(figsize=(2, 2))
    _display_image(X_test[test_ndx], pred=test_pred, actual=test_actual, ax=ax)
    plt.show()

    # show positive train images
    fig, axs = plt.subplots(1, topk_train, figsize=(18, 2))
    axs = axs.flatten()
    for i, (train_ndx, train_impact) in enumerate(pos_impact_list[:topk_train]):
        _display_image(X_train[train_ndx], ax=axs[i], actual=y_train[train_ndx], impact=train_impact)
    plt.tight_layout()
    plt.show()

    # show negative train images
    fig, axs = plt.subplots(1, topk_train, figsize=(18, 2))
    axs = axs.flatten()
    for i, (train_ndx, train_impact) in enumerate(neg_impact_list[:topk_train]):
        _display_image(X_train[train_ndx], ax=axs[i], actual=y_train[train_ndx], impact=train_impact)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', default=5, type=int, help='train subset to use.')
    parser.add_argument('--alpha', default=0.6, type=float, help='transparency of manipulation mask overlay.')
    args = parser.parse_args()
    print(args)
    prediction_explanation(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.topk_train,
                           args.alpha)
