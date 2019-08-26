"""
Instance-based explainer for a tree ensemble using an SVM or LR.
Currently supports: sklearn's RandomForestClassifier and GBMClassifier, lightgbm, xgboost, and catboost.
    Is also only compatible with dense dataset inputs.
"""
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import LabelEncoder

from .extractor import TreeExtractor
from .models.linear_model import SVM, KernelLogisticRegression


class TreeExplainer:

    def __init__(self, tree, X_train, y_train, linear_model='svm', encoding='leaf_output', C=0.1,
                 kernel='linear', gamma='scale', coef0=0.0, degree=3, dense_output=False,
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
        gamma : float (default='scale')
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
            If 'scale', gamma defaults to 1 / (n_features * X.var()).
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
        Computes the decision values for X using the SVM.

        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Returns a 1d array of decision vaues.
        """
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
        assert self.linear_model == 'lr', 'predict_proba only supports lr!'
        return self.linear_model_.predict_proba(self.transform(X))

    def predict(self, X):
        """
        Computes the predicted labels for X using the linear model.

        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Return 1 1d array of predicted labels.
        """
        return self.le.inverse_transform(self.linear_model_.predict(self.transform(X)))

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
            weight = weight.toarray()

        if self.n_classes_ == 2:
            assert weight.shape == (1, self.n_samples_)
        else:
            assert weight.shape == (self.n_classes_, self.n_samples_)

        return weight

    def explain(self, X, y=None):
        """
        Computes the contribution of the training instances on X. A positive number means
        the training instance contributed to the predicted label, and a negative number
        means it contributed away from the predicted label.

        Parameters
        ----------
        X: 2d array-like
            Instances to explain.
        y: 1d array-like (default=None)
            Ground-truth labels of instances being explained; impact scores now represent
            the contribution towards the ground-truth label, instead of the predicted label.
            Must have the same length as X.

        Returns a 2d array of contributions of shape (len(X), n_train_samples). A sparse
            matrix is returned if linear_model=svm and dense_output=False.
        """
        contributions = self.linear_model_.explain(self.transform(X), y=y)
        assert contributions.shape == (len(X), self.n_samples_)

        if self.dense_output and self.linear_model == 'svm':
            contributions = contributions.toarray()

        return contributions

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
        elif 'XGBClassifier' in str(self.tree):
            self.tree_type_ = 'XGBClassifier'
        else:
            exit('{} not currently supported!'.format(str(self.tree)))

        # check linear model
        assert self.linear_model in ['svm', 'lr'], '{} unsupported'.format(self.linear_model)

        # check kernel
        assert self.kernel in ['linear', 'rbf', 'poly', 'sigmoid'], '{} unsupported'.format(self.kernel)

        # check kernel for linear model
        if self.linear_model == 'lr':
            assert self.kernel == 'linear', "lr only supports 'linear' kernel"
