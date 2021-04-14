import numpy as np
import xgboost as xgb


class Bacon:

    def __init__(self, model, X_train, y_train):

        if 'RandomForestClassifier' in str(model):
            self.tree_type = 'rf'
        elif 'CatBoostClassifier' in str(model):
            self.tree_type = 'cb'
        elif 'LGBMClassifier' in str(model):
            self.tree_type = 'lgb'
        elif 'XGBClassifier' in str(model):
            self.tree_type = 'xgb'
        elif 'GradientBoostingClassifier' in str(model):
            self.tree_type = 'gbm'
        else:
            raise ValueError('unknown tree type!')

        self.model = model
        self.X_train = X_train
        self.y_train = y_train

        self.train_leaf_ids_ = self._get_leaf_indices(X_train)

    def get_weights(self, x, y=None):
        x = x.reshape(1, -1)

        instance_leaf_ids = self._get_leaf_indices(x)[0]

        weights = np.zeros(self.X_train.shape[0])
        for i in range(self.train_leaf_ids_.shape[1]):  # per tree
            same_leaf_train_indices = np.where(self.train_leaf_ids_[:, i] == instance_leaf_ids[i])[0]
            if len(same_leaf_train_indices) > 0:
                weights[same_leaf_train_indices] += 1.0 / len(same_leaf_train_indices)

        if y is None:
            pred_label = self.model.predict(x)[0]
            weights = np.where(self.y_train == pred_label, weights, weights * -1)
        else:
            weights = np.where(self.y_train == y, weights, weights * -1)

        return weights

    # private
    def _get_leaf_indices(self, X):
        assert X.ndim == 2
        assert self.tree_type in ['rf', 'cb', 'lgb', 'xgb', 'gbm']

        leaf_ids = None

        if self.tree_type == 'rf':
            leaf_ids = self.model.apply(X)

        elif self.tree_type == 'cb':
            leaf_ids = self.model.calc_leaf_indexes(X)

        elif self.tree_type == 'lgb':
            leaf_ids = self.model.predict_proba(X, pred_leaf=True)

        elif self.tree_type == 'xgb':
            leaf_ids = self.model.get_booster().predict(xgb.DMatrix(X), pred_leaf=True)

        elif self.tree_type == 'gbm':
            leaf_ids = self.model.apply(X).squeeze()

            if leaf_ids.ndim == 1:
                leaf_ids = leaf_ids.reshape(1, -1)

        return leaf_ids
