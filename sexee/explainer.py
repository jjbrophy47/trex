"""
Explainer for a tree ensemble using an SVMs.
Currently supports: sklearn's RandomForestClassifier, lightgbm.
Future support: XGBoost, CatBoost.
"""
import numpy as np
from sklearn.utils.validation import check_X_y
from .extractor import TreeExtractor
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


class TreeExplainer:

    def __init__(self, model, X_train, y_train, encoding='tree_path', random_state=None):

        # error checking
        self._validate_model(model)
        self._validate_data(X_train, y_train)
        assert encoding in ['tree_path', 'tree_output'], '{} encoding unsupported!'.format(encoding)

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.encoding = encoding
        self.random_state = random_state

        # extract feature representations from the tree ensemble
        self.extractor_ = TreeExtractor(self.model, encoding=self.encoding)
        self.train_feature_ = self.extractor_.fit_transform(self.X_train)

        # set kernel for svm
        if self.encoding == 'tree_path':
            self.kernel_ = lambda x, y: np.dot(x, y.T)  # linear kernel

        elif self.encoding == 'tree_output':

            if self.model_type_ == 'RandomForestClassifier':
                self.kernel_ = lambda x, y: np.dot(x, y.T)
            else:
                self.kernel_ = 'rbf'

        # train svm on feature representations and true or predicted labels
        clf = SVC(kernel=self.kernel_, random_state=self.random_state, C=0.1)  # TODO: grid search over C?

        # TODO: option to train on predicted labels?
        if self.n_classes_ > 2:
            self.ovr_ = OneVsRestClassifier(clf).fit(self.train_feature_, self.y_train)
            self.svm_ = None
        else:
            self.svm_ = clf.fit(self.train_feature_, self.y_train)

    def train_impact(self, x, similarity=False, weight=False):
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

        Returns
        -------
        list of (<train_ndx>, <impact>, <sim>, <weight>) tuples for each support vector.
            A positive <impact> score means the support vector contributed towards the predicted label, while a
            negative score means it contributed against the predicted label. <sim> is addded if `similarity`
            is True and <weight> is added if `weight` is True.
        """

        # error checking
        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'
        assert x.ndim == 2, 'x is not 2d!'
        assert x.shape[0] == 1, 'x must be a single instance!'

        # get test instance feature representation
        x_feature = self.extractor_.transform(x)

        # if multiclass, get svm of whose class is predicted
        if self.n_classes_ > 2:
            assert self.ovr_ is not None, 'ovr_ is not fitted!'
            assert self.svm_ is None, 'svm_ already fitted!'
            pred_label = self.ovr_.predict(x_feature)[0]
            self.svm_ = self.ovr_.estimators_[pred_label]
        else:
            assert self.svm_ is not None, 'svm_ is not fitted!'

        # TODO: compute similarity only to support vectors?
        # compute similarity of this instance to all train instances
        sim, sim_ndx = self._similarity(x_feature)

        # ensure the decomposition matches the decision function prediction from the svm
        prediction, impact = self._svm_decomposition(x_feature)
        decision_pred = self.svm_.decision_function(x_feature)[0]
        assert np.isclose(prediction, decision_pred), 'svm.decision_function does not match decomposition!'

        # assemble items to be returned
        impact_list = [self.svm_.support_, impact]
        if similarity:
            impact_list.append(sim[self.svm_.support_])
        if weight:
            impact_list.append(self.svm_.dual_coef_[0])

        result = list(zip(*impact_list))

        # clear chosen svm if multiclass
        if self.n_classes_ > 2:
            self.svm_ = None

        return result

    def _svm_decomposition(self, x_feature):
        """
        Computes the prediction for a query point as a weighted sum of support vectors.
        This should match the `svm.decision_function` method.
        """
        assert x_feature.ndim == 2, 'x_feature is not 2d!'

        sv_feature = self.train_feature_[self.svm_.support_]  # support vector train instances
        sv_weight = self.svm_.dual_coef_[0]  # support vector weights

        sim_prod = np.matmul(sv_feature, x_feature[0])
        weighted_prod = sim_prod * sv_weight
        prediction = (np.sum(weighted_prod) + self.svm_.intercept_)[0]

        return prediction, weighted_prod

    def _similarity(self, x_feature):
        """Finds which instances are most similar to x_feature."""
        # TODO: This is a linear kernel, which only works for `tree_path` encodings!

        assert self.train_feature_ is not None, 'train_feature_ is not fitted!'

        if x_feature.ndim == 2:
            x_feature = x_feature[0]

        sim = np.matmul(self.train_feature_, x_feature)
        sim_ndx = np.argsort(sim)[::-1]

        return sim, sim_ndx

    def _validate_model(self, model):
        """Makes sure the model is a supported model type."""

        model_type = str(model).split('(')[0]
        assert model_type in ['RandomForestClassifier', 'LGBMClassifier'], '{} not supportted!'.format(model_type)
        self.model_type_ = model_type

    def _validate_data(self, X, y):
        check_X_y(X, y)
        if y.dtype == np.float and not np.all(np.mod(y, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_feats_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
