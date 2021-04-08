import numpy as np


class Bacon:

    def __init__(self, model, X_train, y_train):
        assert 'RandomForestClassifier' in str(model) or 'CatBoostClassifier' in str(model)

        if 'RandomForestClassifier' in str(model):
            self.tree_type = 'rf'
        elif 'CatBoostClassifier' in str(model):
            self.tree_type = 'cb'

        self.model = model
        self.X_train = X_train
        self.y_train = y_train

        self.train_leaf_ids_ = self._get_leaf_indices(X_train)

    def get_weights(self, x, y=None):
        x = x.reshape(1, -1)

        instance_leaf_ids = self._get_leaf_indices(x)[0]

        weights = np.zeros(self.X_train.shape[0])
        for i in range(self.train_leaf_ids_.shape[1]):
            same_leaf_train_indices = np.where(self.train_leaf_ids_[:, i] == instance_leaf_ids[i])
            weights[same_leaf_train_indices] += 1.0 / len(same_leaf_train_indices[0])

        if y is None:
            pred_label = self.model.predict(x)[0]
            weights = np.where(self.y_train == pred_label, weights, weights * -1)
        else:
            weights = np.where(self.y_train == y, weights, weights * -1)

        return weights

    def _get_leaf_indices(self, X):
        assert X.ndim == 2

        leaf_ids = None

        if self.tree_type == 'rf':
            leaf_ids = self.model.apply(X)

        elif self.tree_type == 'cb':
            leaf_ids = self.model.calc_leaf_indexes(X)

        return leaf_ids
