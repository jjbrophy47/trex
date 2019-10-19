"""
Exploration: Examine the raw image data from the most impactful train instances for a given test instance.
    Overlay the binary or manipulation mask over the original probe image.
"""
import os
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for TREX

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA

import trex
from utility import model_util, data_util


def _sort_impact(sv_ndx, impact):
    """Sorts support vectors by absolute impact values."""

    if impact.ndim == 2:
        impact = np.sum(impact, axis=1)
    impact_list = zip(sv_ndx, impact)
    impact_list = sorted(sorted(impact_list, key=lambda x: x[0]), key=lambda x: abs(x[1]), reverse=True)

    sv_ndx, impact = zip(*impact_list)
    sv_ndx = np.array(sv_ndx)
    return sv_ndx, impact


def _display_image(x, identifier, predicted, actual, ax=None, impact=None, weight=None, similarity=None, linewidth=3):

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(x.reshape(28, 28), cmap='gray')

    s = identifier
    s += '\n{} predicted as {}'.format(actual, predicted)

    if impact is not None:
        s += '\nimpact: {:.3f}'.format(impact)

    if weight is not None:
        s += '\n' + r'$\alpha$: {:.5f}'.format(weight)

    if similarity is not None:
        s += '\nsimilarity: {:.3f}'.format(similarity)

    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.set_title(s, fontsize=10)


def prediction_explanation(model='lgb', encoding='leaf_output', dataset='mnist_49', n_estimators=100,
                           random_state=69, topk_train=3, test_size=0.1, show_performance=True,
                           true_label=False, pca_components=50, data_dir='data', kernel='linear',
                           linear_model='lr', show_similarity=False, show_weight=False,
                           out_dir='output/mnist'):

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
    explainer = trex.TreeExplainer(tree, X_train_pca, y_train, encoding=encoding, dense_output=True,
                                   use_predicted_labels=not true_label, random_state=random_state,
                                   kernel=kernel, linear_model=linear_model)
    print(explainer)

    if show_performance:
        model_util.performance(tree, X_train_pca, y_train, X_test_pca, y_test)

    # pick a random manipulated test instance to explain
    np.random.seed(random_state)
    test_ndx = np.random.choice(y_test)
    x_test = X_test_pca[test_ndx].reshape(1, -1)
    test_pred = tree.predict(x_test)[0]
    test_actual = y_test[test_ndx]

    # compute the impact of each training instance
    impact = explainer.explain(x_test)[0]
    alpha = explainer.get_weight()[0]
    sim = explainer.similarity(x_test)[0]

    # sort the training instances by impact in descending order
    sort_ndx = np.argsort(impact)[::-1]

    # show the test image
    fig, axs = plt.subplots(1, 1 + topk_train * 2, figsize=(16, 3))
    axs = axs.flatten()
    identifier = 'test_id{}'.format(test_ndx)
    _display_image(X_test[test_ndx], identifier=identifier, predicted=test_pred, actual=test_actual, ax=axs[0])
    plt.setp(axs[0].spines.values(), color='blue')

    # show positive train images
    for i, train_ndx in enumerate(sort_ndx[:topk_train]):
        i += 1
        identifier = 'train_id{}'.format(train_ndx)
        train_pred = tree.predict(X_train_pca[train_ndx].reshape(1, -1))[0]
        similarity = sim[train_ndx] if show_similarity else None
        weight = alpha[train_ndx] if show_weight else None
        plt.setp(axs[i].spines.values(), color='green')
        _display_image(X_train[train_ndx], ax=axs[i], identifier=identifier, predicted=train_pred,
                       actual=y_train[train_ndx], similarity=similarity, weight=weight)

    # show negative train images
    for i, train_ndx in enumerate(sort_ndx[::-1][:topk_train]):
        i += 1 + topk_train
        identifier = 'train_id{}'.format(train_ndx)
        train_pred = tree.predict(X_train_pca[train_ndx].reshape(1, -1))[0]
        similarity = sim[train_ndx] if show_similarity else None
        weight = alpha[train_ndx] if show_weight else None
        plt.setp(axs[i].spines.values(), color='red')
        _display_image(X_train[train_ndx], ax=axs[i], identifier=identifier, predicted=train_pred,
                       actual=y_train[train_ndx], similarity=similarity, weight=weight)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'mnist.pdf'), format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist_49', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in the ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', default=3, type=int, help='train subset to use.')
    parser.add_argument('--true_label', action='store_true', help='train linear model on the true labels.')
    parser.add_argument('--show_similarity', action='store_true', help='Show similarity in the explanation.')
    parser.add_argument('--show_weight', action='store_true', help='Show weight in the explanation.')
    args = parser.parse_args()
    print(args)
    prediction_explanation(model=args.model, encoding=args.encoding, dataset=args.dataset, kernel=args.kernel,
                           n_estimators=args.n_estimators, random_state=args.rs, topk_train=args.topk_train,
                           linear_model=args.linear_model, true_label=args.true_label,
                           show_similarity=args.show_similarity, show_weight=args.show_weight)
