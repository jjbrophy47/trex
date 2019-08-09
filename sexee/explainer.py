"""
Instance-based explainer for a tree ensemble using an SVM or LR.
Currently supports: sklearn's RandomForestClassifier and GBMClassifier, lightgbm, xgboost, and catboost.
    Is also only compatible with dense dataset inputs.
"""
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.preprocessing import LabelEncoder

from .extractor import TreeExtractor
from models.linear_model import SVM, KernelLogisticRegression


class TreeExplainer:

    def __init__(self, tree, X_train, y_train, linear_model='svm', encoding='leaf_output', C=0.1,
                 kernel='linear', gamma=None, coef0=0.0, degree=3, dense_output=False,
                 use_predicted_labels=True, random_state=None):
        """
        Trains an svm on feature representations from a learned tree ensemble.

        Parameters
        ----------
        tree : object
            Learned tree ensemble. Supported: RandomForestClassifier, GBM, LightGBM, CatBoost, XGBoost.
        X_train : 2d array-like
            Train instances in original feature space.
        y_train : 1d array-like (default=None)
            Ground-truth train labels.
        linear_model : str (default='svm', {'svm', 'lr'})
            Linear model to approximate the tree ensemble.
        encoding : str (default='leaf_output', {'leaf_output', 'leaf_path', 'feature_path'})
            Feature representation to extract from the tree ensemble.
        C : float (default=0.1)
            Regularization parameter for the linear model. Lower C values result
            in stronger regularization.
        kernel : str (default='linear', {'linear', 'poly', 'rbf', 'sigmoid'})
            Kernel to use in the dual optimization of the linear model.
            If linear_model='lr', only 'linear' is currently supported.
        gamma : float (default=None)
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
            If None, defaults to 1 / n_features.
            Only applies if linear_model='svm'.
        coef0 : float (default=0.0)
            Independent term in 'poly' and 'sigmoid'.
            Only applies if linear_model='svm'.
        degree : int (default=3)
            Degree of the 'poly' kernel.
            Only applies if linear_model='svm'.
        dense_output : bool (default=False)
            If True, returns impact of all training instances; otherwise, returns
            only support vector impacts and their corresponding training indices.
            Only applies if linear_model='svm'.
        use_predicted_labels : bool (default=True)
            If True, predicted labels from the tree ensemble are used to train the linear model.
        random_state : int (default=None)
            Random state to promote reproducibility.
        """

        # error checking
        self.tree = tree
        self.X_train = X_train
        self.y_train = y_train
        self.linear_model = linear_model
        self.encoding = encoding
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.dense_output = dense_output
        self.use_predicted_labels = use_predicted_labels
        self.random_state = random_state
        self._validate()

        # extract feature representations from the tree ensemble
        self.extractor_ = TreeExtractor(self.tree, encoding=self.encoding)
        self.train_feature_ = self.extractor_.fit_transform(self.X_train)

        # create a linear model to approximate the tree ensemble
        if self.linear_model == 'svm':
            self.linear_model_ = SVM(kernel=self.kernel, C=self.C, gamma=self.gamma,
                                     coef0=self.coef0, degree=self.degree, random_state=self.random_state)
        else:
            self.linear_model_ = KernelLogisticRegression(C=self.C)

        # choose ground truth or predicted labels to train the linear model
        if use_predicted_labels:
            train_label = self.tree.predict(X_train).flatten()
        else:
            train_label = self.y_train

        # encode class labels into numbers between 0 and n_classes - 1
        self.le_ = LabelEncoder().fit(train_label)
        train_label = self.le_.transform(train_label)

        # train the linear model on the tree ensemble feature representation
        self.linear_model_ = self.linear_model_.fit(self.train_feature_, train_label)

    def __str__(self):
        s = '\nTree Explainer:'
        s += '\ntrain shape: {}'.format(self.X_train.shape)
        s += '\nclasses: {}'.format(self.le_.classes_)
        s += '\nlinear_model: {}'.format(self.linear_model_)
        s += '\nencoding: {}'.format(self.encoding)
        s += '\ndense_output: {}'.format(self.dense_output)
        s += '\nfit predicted labels: {}'.format(self.use_predicted_labels)
        s += '\nrandom state: {}'.format(self.random_state)
        s += '\n'
        return s

    def decision_function(self, X):
        """
        Computes the decision value for X using the SVM.

        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Returns a 1d array of decision vaues.
        """
        assert X.ndim == 2, 'X is not 2d!'
        assert self.linear_model == 'svm', 'decision_function only supports svm!'
        return self.linear_model_.decision_function(self.transform(X))

    def predict_proba(self, X):
        """
        Computes the probabilities for X using the logistic regression model.

        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Returns a 2d array of probabilities of shape (len(X), n_classes).
        """
        assert X.ndim == 2, 'X is not 2d!'
        assert self.linear_model == 'lr', 'decision_function only supports lr!'
        return self.linear_model_.predict_proba(self.transform(X))

    def predict(self, X):
        """
        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Return 1 1d array of predicted labels.
        """
        assert X.ndim == 2, 'X is not 2d!'
        return self.linear_model_.predict(self.transform(X))

    # Note: If len(X) is large and the number of training instances is large,
    #       the resulting matrix may be huge.
    def similarity(self, X, train_indices=None):
        """
        Computes the similarity between the instances in X and the training data.

        Parameters
        ----------
        X : 2d array-like
            Instances to compute the similarity to.
        train_indices : 1d array-like (default=None)
            If not None, compute similarities to only these train instances.

        Returns an array of shape (len(X), n_train_samples).
        """
        assert X.ndim == 2, 'X is not 2d!'
        X_sim = self.linear_model_.similarity(self.transform(X), train_indices=train_indices)
        return X_sim

    def get_weight(self):
        """
        Return an array of training instance weights. A sparse output is returned if an
            svm is being used and dense_output=False.
            If binary, returns an array of shape (1, n_train_samples).
            If multiclass, returns an array of shape (n_classes, n_train_samples).
        """
        weight = self.linear_model_.get_weight()

        if self.dense_output and self.linear_model == 'svm':
            weight = np.array(weight.todense())

        if self.n_classes_ == 2:
            assert weight.shape == (1, self.n_samples_)
        else:
            assert weight.shape == (self.n_classes_, self.n_samples_)

        return weight

    def train_impact(self, X, similarity=False, weight=False, intercept=False, y=None):
        """
        Compute the impact of each support vector on one or multiple test instances.
        Currently multiple test instances only supports binary classification problems.

        Parameters
        ----------
        X: 1d or 2d array-like
            Instance to explain in terms of the train instance impact.
        y: 1d array-like (default=None)
            Ground-truth labels of instances being explained; impact scores now represent
            the contribution towrds the ground-truth label, instead of the predicted label.
        similarity: bool (default=False)
            If True, returns the similarity of each support vector to `x`.
        weight: bool (default=False)
            If True, returns the weight of each support vector.
        intercept: bool (default=False)
            If True, returns the intercept of the svm.

        Returns
        -------
        impact_list: tuple of (<train_ndx>, <impact>, <sim>, <weight>, <intercept>).
            A positive impact score means the support vector contributed towards the predicted label, while a
            negative score means it contributed against the predicted (or true) label.
            <sim> is addded if `similarity` is True.
            <weight> is added if `weight` is True.
            <intercept> is added if `intercept` is True.
        """

        # error checking
        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
        X_feature = self.transform(X)

        # if multiclass, get svm of whose class is predicted
        if self.n_classes_ > 2:
            # TODO: implement true label for multiclass
            decision, pred_label = self.decision_function(X, pred_label=True)
            pred_label = int(pred_label[0])
            svm = self.ovr_.estimators_[pred_label]
        else:
            assert self.svm_ is not None, 'svm_ is not fitted!'
            svm = self.svm_
            pred_label = svm.predict(X_feature)

            if y is not None:
                assert len(y) == len(X)
                pred_label = self.le_.transform(y)

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

    # def similarity(self, X, train_indices=None):
    #     """Finds which instances are most similar to each x in X."""

    #     # error checking
    #     assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
    #     X_feature = self.transform(X)

    #     # if multiclass, get svm of whose class is predicted
    #     if self.n_classes_ > 2:
    #         assert X_feature.shape[0] == 1, 'must be 1 instance if n_classes_ > 2!'
    #         decision, pred_label = self.decision_function(X, pred_label=True)
    #         pred_label = int(pred_label[0])
    #         svm = self.ovr_.estimators_[pred_label]
    #     else:
    #         assert self.svm_ is not None, 'svm_ is not fitted!'
    #         svm = self.svm_

    #     # return similarity only for a subset of train instances
    #     if train_indices is not None:
    #         train_feature = self.train_feature_[train_indices]
    #     else:
    #         train_feature = self.train_feature_

    #     # compute similarity
    #     if self.kernel_ == 'linear':
    #         sim = linear_kernel(train_feature, X_feature)
    #         # sim /= self.extractor_.num_trees_
    #     elif self.kernel_ == 'rbf':
    #         sim = rbf_kernel(train_feature, X_feature, gamma=svm._gamma)

    #     return sim

    # def decision_function(self, X, pred_label=False):
    #     """
    #     Parameters
    #     ----------
    #     X : 2d array-like
    #         Instances to make predictions on.
    #     pred_label : bool (default=False)
    #         If True, returns prediction class label in addition to its decision.
    #     Returns decision function values from learned SVM.
    #     If multiclass, returns a flattened array of distances from each SVM.
    #     """
    #     X_feature = self.transform(X)

    #     if self.n_classes_ == 2:
    #         decision = self.svm_.decision_function(X_feature)
    #         pred = np.where(decision < 0, 0, 1)
    #     else:
    #         decision = self.ovr_.decision_function(X_feature)
    #         pred = np.argmax(decision, axis=1)

    #     if pred_label:
    #         pred_class = self.le_.inverse_transform(pred)
    #         result = decision, pred_class
    #     else:
    #         result = decision

    #     return result

    # def predict(self, X):
    #     """Return prediction label for each x in X using the trained SVM."""
    #     decision, pred_label = self.decision_function(X, pred_label=True)
    #     return pred_label

    # def get_train_weight(self, sort=True):
    #     """
    #     Return a list of (train_ndx, weight) tuples for all support vectors.
    #     Currently only supports binary classification.
    #     Parameters
    #     ----------
    #     sorted : bool (default=True)
    #         If True, sorts support vectors by absolute weight value in descending order.
    #     """

    #     assert self.n_classes_ == 2, 'n_classes_ is not 2!'

    #     if self.sparse_:
    #         train_weight = list(zip(self.svm_.support_, np.array(self.svm_.dual_coef_.todense())[0]))
    #     else:
    #         train_weight = list(zip(self.svm_.support_, self.svm_.dual_coef_[0]))

    #     if sort:
    #         train_weight = sorted(train_weight, key=lambda tup: abs(tup[1]), reverse=True)

    #     return train_weight

    def transform(self, X):
        """
        Transform X using the tree-ensemble feature extractor.

        Parameters
        ----------
        X : 2d array-like
            Instances to transform.

        Returns an array of shape (len(X), n_tree_features).
        """
        assert X.ndim == 2, 'X is not 2d!'
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

    def _validate(self):
        """
        Check model and data inputs.
        """

        # check data
        check_X_y(self.X_train, self.y_train)
        if self.y_train.dtype == np.float and not np.all(np.mod(self.y_train, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_samples_ = self.X_train.shape[0]
        self.n_feats_ = self.X_train.shape[1]
        self.classes_ = np.unique(self.y_train)
        self.n_classes_ = len(self.classes_)
        self.labels_ = dict(zip(self.classes_, np.arange(self.n_classes_)))

        # check encoding
        assert self.encoding in ['leaf_path', 'feature_path', 'leaf_output'], '{} unsupported!'.format(self.encoding)

        # check model
        if 'RandomForestClassifier' in str(self.tree):
            self.tree_type_ = 'RandomForestClassifier'
        elif 'GradientBoostingClassifier' in str(self.tree):
            self.tree_type_ = 'GradientBoostingClassifier'
        elif 'LGBMClassifier' in str(self.tree):
            self.tree_type_ = 'LGBMClassifier'
        elif 'CatBoostClassifier' in str(self.tree):
            self.tree_type_ = 'CatBoostClassifier'
        elif 'self.X_trainGBClassifier' in str(self.tree):
            self.tree_type_ = 'XGBClassifier'
        else:
            exit('{} self.tree not currently supported!'.format(str(self.tree)))

        # check linear model
        assert self.linear_model in ['svm', 'lr'], '{} unsupported'.format(self.linear_model)

        # check kernel
        assert self.kernel in ['linear', 'rbf', 'poly', 'sigmoid'], '{} unsupported'.format(self.kernel)
