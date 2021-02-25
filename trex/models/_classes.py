"""
SVM and kernel kernel logistic regression binary classification model
wrappers around Liblinear cython extensions.

NOTE: Liblinear has been modifield to multiply the dual coefficients (alpha)
      by their corresponding training labels y. Thus, alpha values in this
      module can be positive or negative, and multiplying alpha by y is not
      necessary here!
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder

from ._liblinear import train_wrap


class SVM(BaseEstimator, ClassifierMixin):
    """
    Wrapper around liblinear. Solves the l2 regularized l2 loss (squared hinge)
    support vector classifier dual problem using a linear kernel:
        -Solver number 1: https://github.com/cjlin1/liblinear.
        -Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
        -This is equivalent to sklearn's LinearSVC.

    NOTE: Supports binary classification only!
    """

    def __init__(self, C=1.0, pred_size=1000):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter.
        pred_size: int (default=1000)
            Max. number of instancs to predict at one time.
        """
        self.C = C
        self.pred_size = pred_size

    def fit(self, X, y):

        # store training instances for later use
        self.X_train_ = X
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        self.alpha_ = fit_liblinear(X, y, self.C, solver=1)

        return self

    def decision_function(self, X):
        """
        Returns a 1d array of decision values of shape=(no. samples,).
        """
        assert X.ndim == 2

        # concatenate chunks of predictions to avoid memory explosion
        decisions = []
        for i in range(0, len(X), self.pred_size):
            X_sim = linear_kernel(X[i: i + self.pred_size], self.X_train_)
            decisions.append(np.sum(X_sim * self.alpha_, axis=1))

        decision = np.concatenate(decisions)
        return decision

    def predict_proba(self, X):
        """
        Returns a 2d array of probabilities of shape=(no. samples, no. classes).
        """
        assert X.ndim == 2
        a = sigmoid(self.decision_function(X)).reshape(-1, 1)
        return np.hstack([1 - a, a])

    def predict(self, X):
        """
        Returns a 1d array of predicted labels of shape=(no. samples,).
        """
        return np.where(self.decision_function(X) >= 0, 1, 0)

    def compute_attributions(self, x):
        """
        Return a matrix of the impact of the training instances on x.
        The resulting array is of shape (1, n_train_samples).
        """
        assert x.shape == (1, self.X_train_.shape[1])
        x_sim = linear_kernel(x, self.X_train_)
        return x_sim * self.alpha_


class KLR(BaseEstimator, ClassifierMixin):
    """
    Wrapper around liblinear. Solves the l2 logistic
    regression dual problem using a linear kernel:
        -Solver number 7: https://github.com/cjlin1/liblinear.
        -Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
        -This is equivalent to sklearn's LogisticRegression using
         with penalty='l2' and solver='liblinear'.

    NOTE: Supports binary classification only!
    """

    def __init__(self, C=1.0, pred_size=1000):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        pred_size: int (default=1000)
            Max number of instancs to predict at one time. A higher number can
            be faster, but requires more memory to create the similarity matrix.
        """
        self.C = C
        self.pred_size = pred_size

    def fit(self, X, y):

        # store training instances for later use
        self.X_train_ = X
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        self.alpha_ = fit_liblinear(X, y, self.C, solver=7)

        return self

    def predict_proba(self, X):
        """
        Returns a 2d array of probabilities of shape=(X.shape[0], no. classes).
        """
        assert X.ndim == 2

        pos_probas = []
        for i in range(0, len(X), self.pred_size):
            X_sim = linear_kernel(X[i: i + self.pred_size], self.X_train_)
            pos_probas.append(sigmoid(np.sum(X_sim * self.alpha_, axis=1)))
        pos_proba = np.concatenate(pos_probas).reshape(-1, 1)
        proba = np.hstack([1 - pos_proba, pos_proba])
        return proba

    def predict(self, X):
        """
        Returns a 1d array of predicted labels of shape=(X.shape[0],).
        """
        pred_label = np.argmax(self.predict_proba(X), axis=1)
        return pred_label

    def compute_attributions(self, x):
        """
        Return a 2d array of train instance impacts of shape=(1, no train samples).
        """
        assert x.shape == (1, self.X_train_.shape[1])
        x_sim = linear_kernel(x, self.X_train_)
        impact = x_sim * self.alpha_
        return impact


# private
def fit_liblinear(X, y, C, solver, tol=0.1, bias=0, epsilon=0.1):
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

    y_ind = np.where(y_ind == 0, -1, y_ind)

    # LibLinear wants targets as doubles, even for classification
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    y_ind = np.require(y_ind, requirements="W")

    # uniform class weights
    class_weight = np.ones(classes_.shape[0], dtype=np.float64, order='C')

    # liblinear train
    alpha = train_wrap(X,
                       y_ind,
                       solver,
                       tol,
                       bias,
                       C,
                       class_weight,
                       epsilon)

    return alpha[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
