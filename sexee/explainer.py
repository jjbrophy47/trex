"""
Explainer for a tree ensemble using an SVM.
Currently supports: sklearn's RandomForestClassifier, GBMClassifier, lightgbm, xgboost, and catboost.
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

    def __init__(self, model, X_train, y_train, encoding='tree_path', C=0.1, use_predicted_labels=True,
                 random_state=None, timeit=False):
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
        C : float (default=0.1)
            Hyperparameter for the SVM.
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
        self.C = C
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
        clf = SVC(kernel=self.kernel_, random_state=self.random_state, C=self.C, gamma='scale')

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

    def train_impact(self, X, similarity=False, weight=False):
        """
        Compute the impact of each support vector on a single test instance.

        Parameters
        ----------
        1: 1d or 2d array-like
            Instance to explain in terms of the train instance impact.
        similarity: bool
            If True, returns the similarity of each support vector to `x`.
        weight: bool
            If True, returns the weight of each support vector.

        Returns
        -------
        impact_list: list of (<train_ndx>, <impact>, <sim>, <weight>) tuples for each support vector.
            A positive <impact> score means the support vector contributed towards the predicted label, while a
            negative score means it contributed against the predicted label. <sim> is addded if `similarity`
            is True and <weight> is added if `weight` is True.
        """
        start = time.time()

        # error checking
        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
        if X.ndim == 1:
            X = X.reshape(1, -1)
            assert X.shape[0] == 1, 'x must be a single instance!'

        # make a copy to avoid modifying the original
        X = X.copy()

        # get test instance feature representation
        X_feature = self.extractor_.transform(X)

        # if multiclass, get svm of whose class is predicted
        if self.n_classes_ > 2:
            assert self.ovr_ is not None, 'ovr_ is not fitted!'
            assert self.svm_ is None, 'svm_ already fitted!'
            pred_label = int(self.ovr_.predict(X_feature)[0])
            self.svm_ = self.ovr_.estimators_[pred_label]
        else:
            assert self.svm_ is not None, 'svm_ is not fitted!'
            pred_label = self.svm_.predict(X_feature)

        # decompose instance predictions into weighted sums of the train instances
        impact = self._decomposition(X_feature)

        # ensure the decomposition matches the decision function prediction from the svm
        # impact = self._decomposition(X_feature)
        # decision_pred = self.svm_.decision_function(X_feature)[0]
        # assert np.isclose(prediction, decision_pred), 'svm.decision_function does not match decomposition!'

        # flip impact scores for binary case if predicted label is 0
        # this ensures positive impact scores represent contributions toward the predicted label

        dual_weight = self.svm_.dual_coef_[0]
        if self.sparse_:
            dual_weight = np.array(dual_weight.todense())[0]

        # if self.n_classes_ == 2 and pred_label == 0:

        # flip impact scores for binary case if predicted label is 0
        # this ensures positive impact scores represent contributions toward the predicted label
        if self.n_classes_ == 2:
            pred_ndx = np.where(pred_label == 0)[0]
            dual_weight = np.broadcast_to(dual_weight, (len(pred_label), len(dual_weight))).copy()
            dual_weight[pred_ndx] = dual_weight[pred_ndx] * -1
            dual_weight = dual_weight.T
            impact *= -1
            # decision_pred *= -1
            # dual_weight *= -1
        # else:
        #     dual_weight = self.svm_.dual_coef_[0]

        # if self.sparse_:
        #     dual_weight = np.array(dual_weight.todense())[0]

        if X_feature.shape[0] == 1:
            impact = impact.flatten()
            dual_weight = dual_weight.flatten()

        # assemble items to be returned
        impact_tuple = (self.svm_.support_, impact)
        if similarity:
            sim = self.similarity(X)
            if X_feature.shape[0] == 1:
                sim = sim.flatten()
            impact_tuple += (sim[self.svm_.support_],)
        if weight:
            impact_tuple += (dual_weight,)

        # result = list(zip(*impact_list))

        # clear chosen svm if multiclass
        if self.n_classes_ > 2:
            self.svm_ = None

        if self.timeit:
            print('computing impact from train instances took {}s'.format(time.time() - start))

        return impact_tuple

    def similarity(self, X, train_indices=None):
        """Finds which instances are most similar to each x in X."""

        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_feature = self.extractor_.transform(X.copy())
        assert X_feature.shape[1] == self.train_feature_.shape[1], 'num features do not match!'

        if self.n_classes_ > 2:
            assert X.shape[0] == 1, 'must be 1 instance if n_classes_ > 2!'

        if train_indices is not None:
            train_feature = self.train_feature_[train_indices]
        else:
            train_feature = self.train_feature_

        # compute similarity
        if self.kernel_ == 'linear':
            # sim = linear_kernel(train_feature, x_feature).flatten()
            sim = linear_kernel(train_feature, X_feature)
            sim /= self.extractor_.num_trees_
        elif self.kernel_ == 'rbf':
            # sim = rbf_kernel(train_feature, x_feature, gamma=self.svm_._gamma).flatten()
            sim = rbf_kernel(train_feature, X_feature, gamma=self.svm_._gamma)

        result = sim

        # # put train instances in descending order of similarity
        # if sort:
        #     sim_ndx = np.argsort(sim)[::-1]
        #     sim = sim[sim_ndx]
        #     result = (sim, sim_ndx)

        return result

    def get_svm(self):
        """Return a copy of the learned svm model."""

        if self.n_classes_ > 2:
            return copy.deepcopy(self.ovr_)
        else:
            svm_model = copy.deepcopy(self.svm_)

        return svm_model

    def decision_function(self, X, pred_svm=False):
        """
        Return decision function values from learned SVM.
        If multiclass, only supports  a single instance.
        """

        if X.ndim == 1:
            X = X.reshape(1, -1)
            assert X.shape[0] == 1, 'x must be a single instance!'
        assert X.shape[1] == self.n_feats_, 'num features do not match!'

        X_feature = self.extractor_.transform(X)
        assert X_feature.shape[1] == self.train_feature_.shape[1], 'num features do not match!'

        if self.n_classes_ > 2:
            assert X.shape[0] == 1, 'x must be a single instance if n_classes_ > 2!'
            assert self.ovr_ is not None, 'ovr_ is not fitted!'
            pred_label = int(self.ovr_.predict(X_feature)[0])
            svm = self.ovr_.estimators_[pred_label]
            decision = svm.decision_function(X_feature)[0]
        else:
            svm = self.svm_
            pred_label = svm.predict(X_feature)
            decision = svm.decision_function(X_feature)

            # flip distance to separator if binary class and the predicted label is 0
            flip_ndx = np.where(pred_label == 0)[0]
            decision[flip_ndx] = decision[flip_ndx] * -1

            if len(decision) == 1:
                decision = float(decision[0])
            if len(pred_label) == 1:
                pred_label = int(pred_label[0])

        result = decision
        if pred_svm:
            result = decision, pred_label

        return result

    def get_train_weight(self, sort=True):
        """
        Return a list of (train_ndx, weight) tuples for all support vectors.
        Currently only supports binary classification.
        Parameters
        ----------
        sorted : bool (default=True)
            If True, sorts support vectors by absolute weight value in descending order.
        """

        assert self.n_classes_ == 2, 'n_classes_ is not 2!'

        if self.sparse_:
            train_weight = list(zip(self.svm_.support_, np.array(self.svm_.dual_coef_.todense())[0]))
        else:
            train_weight = list(zip(self.svm_.support_, self.svm_.dual_coef_[0]))

        if sort:
            train_weight = sorted(train_weight, key=lambda tup: abs(tup[1]), reverse=True)

        return train_weight

    def _decomposition(self, X_feature):
        """
        Computes the prediction for a query point as a weighted sum of support vectors.
        This should match the `svm.decision_function` method.
        """
        assert X_feature.ndim == 2, 'X_feature is not 2d!'
        if self.n_classes_ > 2:
            assert X_feature.shape[0] == 1, 'X_feature must be 1 instance if n_classes_ > 2!'

        # get support vector instances and weights
        sv_feature = self.train_feature_[self.svm_.support_]  # support vector train instances
        sv_weight = self.svm_.dual_coef_[0]  # support vector weights
        if self.sparse_:
            sv_weight = np.array(sv_weight.todense())[0]
        sv_weight = sv_weight.reshape(-1, 1)

        # compute similarity to the test instance
        if self.kernel_ == 'linear':
            # sim_prod = linear_kernel(sv_feature, X_feature).flatten()
            sim_prod = linear_kernel(sv_feature, X_feature)
        elif self.kernel_ == 'rbf':
            # sim_prod = rbf_kernel(sv_feature, X_feature, gamma=self.svm_._gamma).flatten()
            sim_prod = rbf_kernel(sv_feature, X_feature, gamma=self.svm_._gamma)

        # decompose prediction to a weighted sum of the support vectors
        weighted_prod = sim_prod * sv_weight
        prediction = np.sum(weighted_prod, axis=0) + self.svm_.intercept_[0]

        print(self.svm_.intercept_[0])

        # check to make sure this decomposition is valid
        svm_decision = self.svm_.decision_function(X_feature)
        assert np.allclose(prediction, svm_decision), 'decomposition do not match svm decision!'

        return weighted_prod

    def _validate_data(self, X, y):
        """Make sure the data is well-formed."""

        check_X_y(X, y)
        if y.dtype == np.float and not np.all(np.mod(y, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_feats_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
