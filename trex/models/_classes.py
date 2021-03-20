"""
Surrogate models that identify influential samples for their respective
tree-ensemble models.

SVM and kernel kernel logistic regression binary classification model
wrappers around Liblinear cython extensions.

    NOTE: Liblinear has been modifield to multiply the dual coefficients (alpha)
          by their corresponding training labels y. Thus, alpha values in this
          module can be positive or negative, and multiplying alpha by y is not
          necessary here!

KNN is a wrapper around SKLearn's Kneighbors classifier.

NOTE: Binary classification only!
NOTE: Dense input only!
"""
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder

from ._liblinear import train_wrap


class Surrogate(BaseEstimator, ClassifierMixin):
    """
    Surrogate model abstract class.

    NOTE: Supports binary classification only!
    """

    def __init__(self, tree_extractor, C=1.0, pred_size=1000, random_state=1):
        """
        Parameters
        ----------
        tree_extractor: TreeExtractor object.
            Object to extract features from a tree-ensembel.
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        pred_size: int (default=1000)
            Max number of instancs to predict at one time. A higher number can
            be faster, but requires more memory to create the similarity matrix.
        random_state: int (default=1)
            Random state to seed Liblinear.
        """
        self.tree_extractor = tree_extractor
        self.C = C
        self.pred_size = pred_size
        self.random_state = random_state

    def fit(self, X, y, solver, max_iter=1000):

        # populate attributes
        self.X_train_alt_ = self.tree_extractor.transform(X)
        self.n_features_ = X.shape[1]
        self.n_features_alt_ = self.X_train_alt_.shape[1]
        self.tree_kernel_ = self.tree_extractor.tree_kernel

        # make sure labels are binary
        self.classes_ = np.unique(y)
        assert np.all(self.classes_ == np.array([0, 1]))

        # fit model
        self.alpha_ = fit_liblinear(X=self.X_train_alt_,
                                    y=y,
                                    C=self.C,
                                    solver=solver,
                                    max_iter=max_iter,
                                    random_seed=self.random_state)

        return self

    def predict_proba(self, X):
        """
        Returns a 2d array of probabilities of shape=(X.shape[0], no. classes).
        """
        result_list = []

        # compute probabilities for each chunk of test instances
        for i in range(0, len(X), self.pred_size):
            X_sub_alt = self.transform(X[i: i + self.pred_size])
            X_sub_sim = linear_kernel(X_sub_alt, self.X_train_alt_)
            X_sub_proba = sigmoid(np.sum(X_sub_sim * self.alpha_, axis=1))
            result_list.append(X_sub_proba)

        # assemble result
        pos_proba = np.concatenate(result_list).reshape(-1, 1)
        proba = np.hstack([1 - pos_proba, pos_proba])
        assert proba.shape == (X.shape[0], 2), 'proba shape is no good!'

        return proba

    def predict(self, X):
        """
        Returns a 1d array of predicted labels of shape=(X.shape[0],).
        """
        preds = np.argmax(self.predict_proba(X), axis=1)
        assert preds.shape == (X.shape[0],), 'preds shape is no good!'
        return preds

    def compute_attributions(self, X):
        """
        Compute the attribution of each training sample on each test instance x in X.

        Return a 2d array of train instance impacts of shape=(X.shape[0], no. train samples).
        """
        result_list = []

        # compute attributions of training samples on each input chunk
        for i in range(0, X.shape[0], self.pred_size):
            X_sub_alt = self.transform(X[i: i + self.pred_size])  # 2d
            X_sub_sim = linear_kernel(X_sub_alt, self.X_train_alt_)  # 2d
            X_sub_impact = X_sub_sim * self.alpha_  # 1d
            result_list.append(X_sub_impact)

        # concatenate attributions
        attributions = np.vstack(result_list)
        assert attributions.shape == (X.shape[0], self.X_train_alt_.shape[0]), 'attributions shape is no good!'

        return attributions

    def pred_influence(self, X, pred):
        """
        Compute the attribution of each training sample TOWARDS or AWAY FROM the
        PREDICTED label of each sample x in X; i.e. attributions are flipped
        if the predicted label is 0, otherwise they stay the same.

        Return a 2d array of predicted label influences of shape=(X.shape[0], no. train samples).
        """
        assert X.shape[0] == pred.shape[0], 'no. instances does not match no. labels!'

        # compute attributions
        attributions = self.compute_attributions(X)

        # flip attributions if predicted label is negative
        for i in range(pred.shape[0]):
            if pred[i] == 0:
                attributions[i] *= -1

        # shape check
        assert attributions.shape == (X.shape[0], self.X_train_alt_.shape[0]), 'attributions shape is no good!'

        return attributions

    def similarity(self, X):
        """
        Computes the similarity between the instances in X and the training data.

        Parameters
        ----------
        X : 2D numpy array
            Instances to compute the similarity to.

        Returns an array of shape=(X.shape[0], no. train samples).
        """
        X_alt = self.transform(X)
        X_sim = linear_kernel(X_alt, self.X_train_alt_)
        assert X_sim.shape == (X.shape[0], self.X_train_alt_.shape[0]), 'sim shape is no good!'
        return X_sim

    def transform(self, X):
        """
        Transform X using the tree-ensemble feature extractor.

        Parameters
        ----------
        X : 2d array-like
            Instances to transform.

        Returns an array of shape=(X.shape[0], no. alt. features).
        """
        assert X.ndim == 2, 'X is not 2d!'
        assert X.shape[1] == self.n_features_, 'no. original features do not match!'
        X_alt = self.tree_extractor.transform(X)
        assert X_alt.shape == (X.shape[0], self.n_features_alt_), 'no. transformed features do not match!'
        return X_alt

    def get_alpha(self):
        """
        Return training sample alpha coefficients.
        """
        return self.alpha_.copy()


class KLR(Surrogate):
    """
    Wrapper around liblinear. Solves the l2 logistic
    regression dual problem using a linear kernel:
        -Solver number 7: https://github.com/cjlin1/liblinear.
        -Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
        -This is equivalent to sklearn's LogisticRegression using
         with penalty='l2' and solver='liblinear'.

    NOTE: Supports binary classification only!
    """

    # override
    def fit(self, X, y):
        """
        Fit model using Liblinear solver no. 7.
        """
        return super().fit(X, y, solver=7)


class SVM(Surrogate):
    """
    Wrapper around liblinear. Solves the l2 regularized l2 loss (squared hinge)
    support vector classifier dual problem using a linear kernel:
        -Solver number 1: https://github.com/cjlin1/liblinear.
        -Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
        -This is equivalent to sklearn's LinearSVC.

    NOTE: Supports binary classification only!
    """

    # overrride
    def fit(self, X, y):
        """
        Fit model using Liblinear solver no. 1.
        """
        return super().fit(X, y, solver=1)

    # extend
    def decision_function(self, X):
        """
        Returns a 1d array of decision values of shape=(no. samples,).
        """
        result_list = []

        # concatenate chunks of predictions to avoid memory explosion
        for i in range(0, len(X), self.pred_size):
            X_sub_alt = self.transform(X[i: i + self.pred_size])  # 2d
            X_sub_sim = linear_kernel(X_sub_alt, self.X_train_alt_)  # 2d
            result_list.append(np.sum(X_sub_sim * self.alpha_, axis=1))  # 1d

        # concatenate decisions
        decision_values = np.concatenate(result_list)
        assert decision_values.shape == (X.shape[0],), 'decision_values shape is no good!'

        return decision_values

    # override
    def predict_proba(self, X):
        """
        Returns a 2d array of probabilities of shape=(no. samples, no. classes).
        """
        pos_proba = sigmoid(self.decision_function(X)).reshape(-1, 1)
        probas = np.hstack([1 - pos_proba, pos_proba])
        assert probas.shape == (X.shape[0], 2), 'probas shape is no good!'
        return probas

    # override
    def predict(self, X):
        """
        Returns a 1d array of predicted labels of shape=(no. samples,).
        """
        preds = np.where(self.decision_function(X) >= 0, 1, 0)
        assert preds.shape == (X.shape[0],), 'preds shape is no good!'
        return preds


class KNN(KNeighborsClassifier):
    """
    Wrapper around SKLearn's KNeighborsClassifier that takes care of
    transforming the data before predicting.

    NOTE: Supports binary classification only!
    """

    # override
    def __init__(self, tree_extractor, n_neighbors=5, weights='uniform'):
        """
        Fit model using Liblinear solver no. 7.
        """
        self.tree_extractor = tree_extractor
        return super().__init__(n_neighbors=n_neighbors, weights=weights)

    # override
    def fit(self, X, y):
        """
        Transform using tree extractor, and then call `fit`.
        """

        # populate attributes
        self.X_train_alt_ = self.tree_extractor.transform(X)
        self.y_train_ = y.copy()
        self.n_features_ = X.shape[1]
        self.n_features_alt_ = self.X_train_alt_.shape[1]
        self.tree_kernel_ = self.tree_extractor.tree_kernel

        return super().fit(self.X_train_alt_, y)

    # override
    def predict_proba(self, X):
        """
        Transform using tree extractor, and then call `predict_proba`.
        """
        return super().predict_proba(X)

    # override
    def predict(self, X):
        """
        Transform using tree extractor, and then call `predict`.
        """
        preds = np.argmax(self.predict_proba(X), axis=1)
        assert preds.shape == (X.shape[0],), 'preds shape is no good!'
        return preds

    # override
    def kneighbors(self, X):
        """
        Transform using tree extractor, and then call `kneighbors`.
        """
        X_alt = self.transform(X)
        return super().kneighbors(X_alt)

    def compute_attributions(self, X, start_pred=None, frac_progress=0.1, logger=None):
        """
        Return a 2d array of train instance impacts of shape=(X.shape[0], no. train samples).
        """
        start = time.time()

        # result container
        attributions = np.zeros((X.shape[0], self.X_train_alt_.shape[0]))

        # compute the contribution of all training samples on each test instance
        for i in range(X.shape[0]):
            x = X[[i]]

            # get predicted label and neighbors
            distances, neighbor_ids = self.kneighbors(x)

            # add density to neighbor training instances if they have the same label as the predicted label
            for neighbor_id in neighbor_ids[0]:

                # gives positive weight to excitatory instances (w.r.t. to the predicted label)
                if start_pred is None:
                    pred = int(self.predict(x)[0])
                    attributions[i][neighbor_id] = 1 if self.y_train_[neighbor_id] == pred else -1

                # gives positive weight to excitatory instances (w.r.t. to the given label)
                else:
                    attributions[i][neighbor_id] = 1 if self.y_train_[neighbor_id] == start_pred else -1

            # display progress
            if logger and i % int(X.shape[0] * frac_progress) == 0:
                elapsed = time.time() - start
                logger.info('done {:.1f}% test instances...{:.3f}s'.format((i / X.shape[0]) * 100, elapsed))

        return attributions

    def transform(self, X):
        """
        Transform X using the tree-ensemble feature extractor.

        Parameters
        ----------
        X : 2d array-like
            Instances to transform.

        Returns an array of shape=(X.shape[0], no. alt. features).
        """
        assert X.ndim == 2, 'X is not 2d!'
        assert X.shape[1] == self.n_features_, 'no. original features do not match!'
        X_alt = self.tree_extractor.transform(X)
        assert X_alt.shape == (X.shape[0], self.n_features_alt_), 'no. transformed features do not match!'
        return X_alt


# private
def fit_liblinear(X, y, C, solver, eps=0.1, bias=0, max_iter=1000,
                  random_seed=1, epsilon=0.1, class_weight=None, sample_weight=None,
                  is_sparse=False):
    """
    Used by Logistic Regression (and CV) and LinearSVC/LinearSVR.
    Preprocessing is done in this function before supplying it to liblinear.

    Adapted from:
    https://github.com/scikit-learn/scikit-learn/blob/\
    95119c13af77c76e150b753485c662b7c52a41a2/sklearn/svm/_base.py#L835

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.
    y : array-like of shape (n_samples,)
        Target vector relative to X
    C : float
        Lower C = more penalization.
    Solver : str (default='1')
        '1' for Linear SVC (dual) with L2 regularization.
        '7' for LR (dual) with L2 regularization.

    Returns
    -------
    alpha : ndarray of shape (no. samples,)
        Coefficients of all training samples.
    """
    enc = LabelEncoder()
    y_ind = enc.fit_transform(y)
    classes_ = enc.classes_

    # stores a new array in case a slice is passed in
    X = X.copy()

    # LibLinear wants targets as doubles, even for classification
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    y_ind = np.require(y_ind, requirements="W")

    # set class weights
    if class_weight is None:
        class_weight = np.ones(classes_.shape[0], dtype=np.float64, order='C')
    assert class_weight.shape[0] == classes_.shape[0]

    # set sample weights
    if sample_weight is None:
        sample_weight = np.ones(y_ind.shape[0], dtype=np.float64, order='C')
    assert sample_weight.shape[0] == y_ind.shape[0]

    # liblinear train
    alpha = train_wrap(X,
                       y_ind,
                       is_sparse,
                       solver,
                       eps,
                       bias,
                       C,
                       class_weight,
                       max_iter,
                       random_seed,
                       epsilon,
                       sample_weight)

    return alpha[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
