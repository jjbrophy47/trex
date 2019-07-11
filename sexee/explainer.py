"""
Explainer for a tree ensemble using an SVM.
Currently supports: sklearn's RandomForestClassifier, GBMClassifier, lightgbm, xgboost, and catboost.
"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_X_y
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.preprocessing import LabelEncoder

from . import util
from .extractor import TreeExtractor


class TreeExplainer:

    def __init__(self, model, X_train, y_train, encoding='leaf_output', C=0.1, gamma='scale',
                 use_predicted_labels=True, random_state=None, timeit=False):
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
        encoding : str (default='leaf_path')
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
        assert encoding in ['leaf_path', 'feature_path', 'leaf_output'], '{} unsupported!'.format(encoding)

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.encoding = encoding
        self.C = C
        self.gamma = gamma
        self.use_predicted_labels = use_predicted_labels
        self.random_state = random_state
        self.timeit = timeit

        # set kernel for svm
        if self.encoding == 'leaf_path' or self.encoding == 'feature_path':
            self.kernel_ = 'linear'
            self.sparse_ = True

        elif self.encoding == 'leaf_output':

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

        # train svm on feature representations and labels (true or predicted)
        clf = SVC(kernel=self.kernel_, random_state=self.random_state, C=self.C, gamma=self.gamma)

        # choose ground truth or predicted labels to train the svm
        if use_predicted_labels:
            train_label = self.model.predict(X_train).flatten()
        else:
            train_label = self.y_train

        # encode class labels into numbers between 0 and n_classes - 1
        self.le_ = LabelEncoder().fit(train_label)
        train_label = self.le_.transform(train_label)

        # if multiclass then train an SVM for each class, otherwise train one SVM
        start = time.time()
        if self.n_classes_ > 2:
            self.ovr_ = OneVsRestClassifier(clf).fit(self.train_feature_, train_label)
            self.svm_ = None
        else:
            self.svm_ = clone(clf).fit(self.train_feature_, train_label)
        if self.timeit:
            print('fitting svm took {}s'.format(time.time() - start))

    def __str__(self):
        s = '\nTree Explainer:'
        s += '\ntrain shape: {}'.format(self.X_train.shape)
        s += '\nclasses: {}'.format(self.le_.classes_)
        s += '\nencoding: {}'.format(self.encoding)
        s += '\nsparse: {}'.format(self.sparse_)
        s += '\nkernel: {}'.format(self.kernel_)
        s += '\nfit predicted labels: {}'.format(self.use_predicted_labels)
        s += '\nC: {}'.format(self.C)
        s += '\ngamma: {}'.format(self.gamma)
        s += '\nrandom state: {}'.format(self.random_state)
        s += '\n'
        return s

    def train_impact(self, X, similarity=False, weight=False, intercept=False):
        """
        Compute the impact of each support vector on one or multiple test instances.
        Currently multiple test instances only supports binary classification problems.

        Parameters
        ----------
        X: 1d or 2d array-like
            Instance to explain in terms of the train instance impact.
        similarity: bool
            If True, returns the similarity of each support vector to `x`.
        weight: bool
            If True, returns the weight of each support vector.
        intercept: bool
            If True, returns the intercept of the svm.

        Returns
        -------
        impact_list: tuple of (<train_ndx>, <impact>, <sim>, <weight>, <intercept>).
            A positive impact score means the support vector contributed towards the predicted label, while a
            negative score means it contributed against the predicted label.
            <sim> is addded if `similarity` is True.
            <weight> is added if `weight` is True.
            <intercept> is added if `intercept` is True.
        """

        # error checking
        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
        X_feature = self.transform(X)

        # if multiclass, get svm of whose class is predicted
        if self.n_classes_ > 2:
            decision, pred_label = self.decision_function(X, pred_label=True)
            pred_label = int(pred_label[0])
            svm = self.ovr_.estimators_[pred_label]
        else:
            assert self.svm_ is not None, 'svm_ is not fitted!'
            svm = self.svm_
            pred_label = svm.predict(X_feature)

        # decompose instance predictions into weighted sums of the train instances
        impact = self._decomposition(X_feature, svm=svm)

        # get support vector weights
        dual_weight = svm.dual_coef_[0].copy()
        if self.sparse_:
            dual_weight = np.array(dual_weight.todense())[0]

        # get intercept
        svm_intercept = svm.intercept_[0]

        # flip impact scores for binary case if predicted label is 0
        # this ensures positive impact scores represent contributions toward the predicted label
        if self.n_classes_ == 2:
            dual_weight = np.broadcast_to(dual_weight, (len(pred_label), len(dual_weight))).copy()

            pred_ndx = np.where(pred_label == 0)[0]
            dual_weight[pred_ndx] = dual_weight[pred_ndx] * -1
            dual_weight = dual_weight.T
            impact[:, pred_ndx] *= -1
            svm_intercept *= -1

        # return a 1d array if a single instance is given
        if X_feature.shape[0] == 1:
            impact = impact.flatten()
            dual_weight = dual_weight.flatten()

        # assemble items to be returned
        impact_tuple = (svm.support_, impact)

        if similarity:
            sim = self.similarity(X)
            if X_feature.shape[0] == 1:
                sim = sim.flatten()
            impact_tuple += (sim[svm.support_],)

        if weight:
            impact_tuple += (dual_weight,)

        if intercept:
            impact_tuple += (svm_intercept,)

        return impact_tuple

    def similarity(self, X, train_indices=None):
        """Finds which instances are most similar to each x in X."""

        # error checking
        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
        X_feature = self.transform(X)

        # if multiclass, get svm of whose class is predicted
        if self.n_classes_ > 2:
            assert X_feature.shape[0] == 1, 'must be 1 instance if n_classes_ > 2!'
            decision, pred_label = self.decision_function(X, pred_label=True)
            pred_label = int(pred_label[0])
            svm = self.ovr_.estimators_[pred_label]
        else:
            assert self.svm_ is not None, 'svm_ is not fitted!'
            svm = self.svm_

        # return similarity only for a subset of train instances
        if train_indices is not None:
            train_feature = self.train_feature_[train_indices]
        else:
            train_feature = self.train_feature_

        # compute similarity
        if self.kernel_ == 'linear':
            sim = linear_kernel(train_feature, X_feature)
            # sim /= self.extractor_.num_trees_
        elif self.kernel_ == 'rbf':
            sim = rbf_kernel(train_feature, X_feature, gamma=svm._gamma)

        return sim

    def decision_function(self, X, pred_label=False):
        """
        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.
        pred_label : bool (default=False)
            If True, returns prediction class label in addition to its decision.
        Returns decision function values from learned SVM.
        If multiclass, returns a flattened array of distances from each SVM.
        """
        X_feature = self.transform(X)

        if self.n_classes_ == 2:
            decision = self.svm_.decision_function(X_feature)
            pred = np.where(decision < 0, 0, 1)
        else:
            decision = self.ovr_.decision_function(X_feature)
            pred = np.argmax(decision, axis=1)

        if pred_label:
            pred_class = self.le_.inverse_transform(pred)
            result = decision, pred_class
        else:
            result = decision

        return result

    def predict(self, X):
        """Return prediction label for each x in X using the trained SVM."""
        decision, pred_label = self.decision_function(X, pred_label=True)
        return pred_label

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

    def transform(self, X):
        """Transform X using the extractor."""

        if X.ndim == 1:
            X = X.reshape(1, -1)
            assert X.shape[0] == 1, 'x must be a single instance!'
        assert X.shape[1] == self.n_feats_, 'num features do not match!'

        X_feature = self.extractor_.transform(X)
        assert X_feature.shape[1] == self.train_feature_.shape[1], 'num features do not match!'

        return X_feature

    def _decomposition(self, X_feature, svm):
        """
        Computes the prediction for a query point as a weighted sum of support vectors.
        This should match the `svm.decision_function` method.
        """
        assert X_feature.ndim == 2, 'X_feature is not 2d!'
        if self.n_classes_ > 2:
            assert X_feature.shape[0] == 1, 'X_feature must be 1 instance if n_classes_ > 2!'

        # get support vector instances and weights
        sv_feature = self.train_feature_[svm.support_]  # support vector train instances
        sv_weight = svm.dual_coef_[0]  # support vector weights
        if self.sparse_:
            sv_weight = np.array(sv_weight.todense())[0]
        sv_weight = sv_weight.reshape(-1, 1)

        # compute similarity to the test instance
        if self.kernel_ == 'linear':
            sim_prod = linear_kernel(sv_feature, X_feature)
        elif self.kernel_ == 'rbf':
            sim_prod = rbf_kernel(sv_feature, X_feature, gamma=svm._gamma)

        # decompose prediction to a weighted sum of the support vectors
        weighted_prod = sim_prod * sv_weight
        prediction = np.sum(weighted_prod, axis=0) + svm.intercept_[0]

        # check to make sure this decomposition is valid
        svm_decision = svm.decision_function(X_feature)
        assert np.allclose(prediction, svm_decision), 'decomposition does not match svm decision!'

        return weighted_prod

    def _validate_data(self, X, y):
        """Make sure the data is well-formed."""

        check_X_y(X, y)
        if y.dtype == np.float and not np.all(np.mod(y, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_feats_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.labels_ = dict(zip(self.classes_, np.arange(self.n_classes_)))
