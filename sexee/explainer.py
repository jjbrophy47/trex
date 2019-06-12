"""
Explainer for a tree ensemble using an SVMs.
Currently supports: sklearn's RandomForestClassifier, lightgbm.
Future support: XGBoost, CatBoost.
"""
import time
import copy

import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

from . import util
from .extractor import TreeExtractor


class TreeExplainer:

    def __init__(self, model, X_train, y_train, encoding='tree_path', use_predicted_labels=True, random_state=None,
                 timeit=False):
        """
        Trains an svm on feature representations from a learned tree ensemble.

        Parameters
        ----------
        model : object
            Learned tree ensemble. Supported: RandomForestClassifier, LightGBM, CatBoost.
            Unsupported: XGBoost, GBM.
        X_train : 2d array-like
            Train instances in original feature space.
        y_train : 1d array-like (default=None)
            Ground-truth train labels.
        encoding : str (default='tree_path')
            Feature representation to extract from the tree ensemble.
        use_predicted_labels : bool (default=True)
            If True, predicted labels from the tree ensemble are used to train the SVM.
        random_state : int (default=None)
            Random state to promote reproducibility.
        timeit : bool (default=False)
            Displays feature extraction and SVM fit times.
        """

        # error checking
        self.model_type_ = util.validate_model(model)
        self._validate_data(X_train, y_train)
        assert encoding in ['tree_path', 'tree_output'], '{} encoding unsupported!'.format(encoding)

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.encoding = encoding
        self.use_predicted_labels = use_predicted_labels
        self.random_state = random_state
        self.timeit = timeit

        # set kernel for svm
        if self.encoding == 'tree_path':
            self.kernel_ = 'linear'
            self.sparse_ = True

        elif self.encoding == 'tree_output':

            if self.model_type_ == 'RandomForestClassifier':
                self.kernel_ = 'linear'
                self.sparse_ = True
            else:
                self.kernel_ = 'rbf'
                self.sparse_ = False

        # extract feature representations from the tree ensemble
        start = time.time()
        self.extractor_ = TreeExtractor(self.model, encoding=self.encoding, sparse=self.sparse_)
        self.train_feature_ = self.extractor_.fit_transform(self.X_train)
        if self.timeit:
            print('train feature extraction took {}s'.format(time.time() - start))

        # train svm on feature representations and true or predicted labels
        # TODO: grid search over C?
        # TODO: grid search over gamma when rbf kernel?
        clf = SVC(kernel=self.kernel_, random_state=self.random_state, C=0.5, gamma='scale')

        # choose ground truth or predicted labels to train the svm on
        if use_predicted_labels:
            train_label = self.model.predict(X_train)
        else:
            train_label = self.y_train

        # train `n_classes_` SVM models if multiclass, otherwise just train one SVM
        start = time.time()
        if self.n_classes_ > 2:
            self.ovr_ = OneVsRestClassifier(clf).fit(self.train_feature_, train_label)
            self.svm_ = None
        else:
            self.svm_ = clf.fit(self.train_feature_, train_label)
        if self.timeit:
            print('fitting svm took {}s'.format(time.time() - start))

    def train_impact(self, x, similarity=False, weight=False, pred_svm=False):
        """
        Compute the impact of each support vector on a single test instance.

        Parameters
        ----------
        x: 2d array-like
            Instance to explain in terms of the train instance impact.
        similarity: bool
            If True, returns the similarity of each support vector to `x`.
        weight: bool
            If True, returns the weight of each support vector.
        pred_svm: bool
            If True, returns an svm_pred: (<distance to separator>, <predicted_label>) tuple from the svm.

        Returns
        -------
        impact_list: list of (<train_ndx>, <impact>, <sim>, <weight>) tuples for each support vector.
            A positive <impact> score means the support vector contributed towards the predicted label, while a
            negative score means it contributed against the predicted label. <sim> is addded if `similarity`
            is True and <weight> is added if `weight` is True.
        If `pred_svm` is True, the return object becomes (impact_list, svm_pred) tuple.
        """
        start = time.time()

        # error checking
        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
        assert x.ndim == 2, 'x is not 2d!'
        assert x.shape[0] == 1, 'x must be a single instance!'

        # get test instance feature representation
        x_feature = self.extractor_.transform(x.copy())

        # if multiclass, get svm of whose class is predicted
        if self.n_classes_ > 2:
            assert self.ovr_ is not None, 'ovr_ is not fitted!'
            assert self.svm_ is None, 'svm_ already fitted!'
            pred_label = int(self.ovr_.predict(x_feature)[0])
            self.svm_ = self.ovr_.estimators_[pred_label]
        else:
            assert self.svm_ is not None, 'svm_ is not fitted!'
            pred_label = int(self.svm_.predict(x_feature)[0])

        # compute similarity of this instance to all train instances
        sim = self.similarity(x_feature)

        # ensure the decomposition matches the decision function prediction from the svm
        prediction, impact = self._decomposition(x_feature)
        decision_pred = self.svm_.decision_function(x_feature)[0]
        assert np.isclose(prediction, decision_pred), 'svm.decision_function does not match decomposition!'

        # flip impact scores for binary case if predicted label is 0
        # this ensures positive impact scores represent contributions toward the predicted label
        if self.n_classes_ == 2 and pred_label == 0:
            impact *= -1
            decision_pred *= -1
            dual_weight = self.svm_.dual_coef_[0] * -1
        else:
            dual_weight = self.svm_.dual_coef_[0]

        if self.sparse_:
            dual_weight = np.array(dual_weight.todense())[0]

        # assemble items to be returned
        impact_list = [self.svm_.support_, impact]
        if similarity:
            impact_list.append(sim[self.svm_.support_])
        if weight:
            impact_list.append(dual_weight)

        result = list(zip(*impact_list))

        if pred_svm:
            result = (result, (decision_pred, pred_label))

        # clear chosen svm if multiclass
        if self.n_classes_ > 2:
            self.svm_ = None

        if self.timeit:
            print('computing impact from train instances took {}s'.format(time.time() - start))

        return result

    def similarity(self, x_feature, sort=False):
        """Finds which instances are most similar to x_feature."""

        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'

        # compute similarity
        if self.kernel_ == 'linear':
            sim = linear_kernel(self.train_feature_, x_feature).flatten()
        elif self.kernel_ == 'rbf':
            sim = rbf_kernel(self.train_feature_, x_feature, gamma=self.svm_._gamma).flatten()

        result = sim

        # put train instances in descending order of similarity
        if sort:
            sim_ndx = np.argsort(sim)[::-1]
            result = (sim, sim_ndx)

        return result

    def get_svm(self):
        """Return a copy of the learned svm model."""

        if self.n_classes_ > 2:
            return copy.deepcopy(self.ovr_)
        else:
            svm_model = copy.deepcopy(self.svm_)

        return svm_model

    def _decomposition(self, x_feature):
        """
        Computes the prediction for a query point as a weighted sum of support vectors.
        This should match the `svm.decision_function` method.
        """
        assert x_feature.ndim == 2, 'x_feature is not 2d!'

        # get support vector instances and weights
        sv_feature = self.train_feature_[self.svm_.support_]  # support vector train instances
        sv_weight = self.svm_.dual_coef_[0]  # support vector weights
        if self.sparse_:
            sv_weight = np.array(sv_weight.todense())[0]

        # compute similarity to the test instance
        if self.kernel_ == 'linear':
            sim_prod = linear_kernel(sv_feature, x_feature).flatten()
        elif self.kernel_ == 'rbf':
            sim_prod = rbf_kernel(sv_feature, x_feature, gamma=self.svm_._gamma).flatten()

        # decompose prediction to a weighted sum of the support vectors
        weighted_prod = sim_prod * sv_weight
        prediction = (np.sum(weighted_prod) + self.svm_.intercept_)[0]

        return prediction, weighted_prod

    def _validate_data(self, X, y):
        """Make sure the data is well-formed."""

        check_X_y(X, y)
        if y.dtype == np.float and not np.all(np.mod(y, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_feats_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
