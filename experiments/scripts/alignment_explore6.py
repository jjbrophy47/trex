"""
Exploration: Examine the raw image data from the most impactful train instances for a given test instance.
    Overlay the binary or manipulation mask over the original probe image.
"""
import os
import argparse

import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from PIL import Image, ExifTags

from util import model_util, data_util, exp_util, print_util


def _sort_impact(sv_ndx, impact):
    """Sorts support vectors by absolute impact values."""

    if impact.ndim == 2:
        impact = np.sum(impact, axis=1)
    impact_list = zip(sv_ndx, impact)
    impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)

    sv_ndx, impact = zip(*impact_list)
    sv_ndx = np.array(sv_ndx)
    return sv_ndx, impact


def _white_to_transparency(img):
    from PIL import Image

    x = np.asarray(img.convert('RGBA')).copy()
    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
    return Image.fromarray(x)


def _open_image(image_name, to_rgb=False, transparent_white=False):

    img = Image.open(image_name)

    if to_rgb:
        img = img.convert('RGB')

    # rotate image if exif data says to: https://stackoverflow.com/a/11543365
    if hasattr(img, '_getexif'):  # only present in JPEGs

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        e = img._getexif()       # returns None if no EXIF data
        if e is not None:
            exif = dict(e.items())

            if orientation in exif:
                orientation = exif[orientation]
                if orientation == 3:
                    img = img.rotate(180)
                elif orientation == 6:
                    img = img.rotate(270)
                elif orientation == 8:
                    img = img.rotate(90)

    if transparent_white:
        img = _white_to_transparency(img)

    return img


def _display_image(image_id, image_ref, ax=None, alpha=0.6, pred=None, actual=None, impact=None, data_dir='data'):

    if ax is None:
        fig, ax = plt.subplots()

    image_info = image_ref[image_ref['image_id'] == image_id]
    probe_fn = image_info['ProbeFileName'].values[0].split('/')[-1]
    probe_img = _open_image(os.path.join(data_dir, 'MFC18_EvalPart1', 'probe', probe_fn))
    ax.imshow(probe_img)

    mask_fn = image_info['ProbeMaskFileName'].values[0]
    if isinstance(mask_fn, str):
        mask_fn = str(mask_fn).split('/')[-1]
        mask_img = _open_image(os.path.join(data_dir, 'MFC18_EvalPart1', 'manipulation_mask', mask_fn),
                               to_rgb=True, transparent_white=True)
        mask_img = _white_to_transparency(mask_img)
        ax.imshow(mask_img, alpha=alpha)

    s = ''

    if pred is not None:
        s += 'predicted: {}\n'.format(pred)

    if actual is not None:
        s += 'label: {}'.format(actual)

    if impact is not None:
        s += '\nimpact: {:.3f}'.format(impact)

    ax.axis('off')
    ax.set_title(s, fontsize=11)


def prediction_explanation(model='lgb', encoding='tree_path', dataset='medifor1b', n_estimators=100, random_state=69,
                           topk_train=5, test_size=0.1, alpha=0.5, show_performance=True, data_dir='data'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_image_id=True,
                              test_size=test_size)
    X_train, X_test, y_train, y_test, id_train, id_test, label = data

    # fit a tree ensemble and an explainer for that tree ensemble
    tree = clone(clf).fit(X_train, y_train)
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding)

    if show_performance:
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    # pick a random manipulated test instance to explain
    y_test_manip = np.where(y_test == 1)[0]
    np.random.seed(random_state)
    test_ndx = np.random.choice(y_test_manip)
    x_test = X_test[test_ndx]
    test_pred = label[tree.predict(x_test.reshape(1, -1))[0]]
    test_actual = label[y_test[test_ndx]]

    sv_ndx, impact = explainer.train_impact(x_test)
    sv_ndx, impact = _sort_impact(sv_ndx, impact)

    # print impactful train instances
    impact_list = list(zip(sv_ndx, impact))
    pos_impact_list = [impact_item for impact_item in impact_list if impact_item[1] > 0]
    neg_impact_list = [impact_item for impact_item in impact_list if impact_item[1] < 0]
    svm_pred, pred_label = explainer.decision_function(x_test, pred_svm=True)

    import os
    import pandas as pd

    # image_ref = pd.read_csv('data/MFC18_EvalPart1/image_ref.csv')

    # TODO: put this into the data_utils module
    image_ref = pd.read_csv(os.path.join(data_dir, 'MFC18_EvalPart1/image_ref.csv'))

    # show the test image
    _display_image(id_test[test_ndx], image_ref, alpha=alpha, pred=test_pred, actual=test_actual, data_dir=data_dir)
    plt.show()

    # show positive train images
    fig, axs = plt.subplots(1, topk_train, figsize=(18, 4))
    axs = axs.flatten()
    for i, (train_ndx, train_impact) in enumerate(pos_impact_list[:topk_train]):
        _display_image(id_train[train_ndx], image_ref, alpha=alpha, ax=axs[i], actual=label[y_train[train_ndx]],
                       data_dir=data_dir, impact=train_impact)
    plt.tight_layout()
    plt.show()

    # show negative train images
    fig, axs = plt.subplots(1, topk_train, figsize=(18, 4))
    axs = axs.flatten()
    for i, (train_ndx, train_impact) in enumerate(neg_impact_list[:topk_train]):
        _display_image(id_train[train_ndx], image_ref, alpha=alpha, ax=axs[i], actual=label[y_train[train_ndx]],
                       data_dir=data_dir, impact=train_impact)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medifor1b', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', default=5, type=int, help='train subset to use.')
    parser.add_argument('--test_size', default=0.2, type=float, help='size of the test set.')
    parser.add_argument('--alpha', default=0.6, type=float, help='transparency of manipulation mask overlay.')
    args = parser.parse_args()
    print(args)
    prediction_explanation(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.topk_train,
                           args.test_size, args.alpha)
