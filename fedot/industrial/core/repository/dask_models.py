from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
from dask_ml.linear_model import LogisticRegression, LinearRegression
from dask_ml.decomposition import PCA
import numpy as np
import dask.array as da


class DaskLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, params):
        """
        Custom estimator based on Dask LogisticRegression.
        """
        self.penalty = params.get('penalty', 'l2')
        self.C = params.get('C', 1.0)
        self.model_ = None  # Placeholder for the internal Dask model
        self.solver = 'admm'

    def fit(self, X, y):
        """
        Fit the model using Dask's LogisticRegression.
        """

        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        self.classes_ = np.unique(y)
        if not isinstance(X, da.Array):
            X = da.from_array(X)
        if not isinstance(y, da.Array):
            y = da.from_array(y)

        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        X = check_array(X, accept_sparse=True, dtype=None)
        if not isinstance(X, da.Array):
            X = da.from_array(X)
        return self.model_.predict(X).compute()

    def predict_proba(self, X):
        """
        Predict probabilities for samples in X.
        """
        X = check_array(X, accept_sparse=True, dtype=None)
        if not isinstance(X, da.Array):
            X = da.from_array(X)
        return self.model_.predict_proba(X).compute()

    def score(self, X, y):
        """
        Returns the accuracy of the model.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        """
        Return hyperparameter dictionary for compatibility with GridSearchCV.
        """
        return {
            "penalty": self.penalty,
            "C": self.C,
        }

    def set_params(self, **params):
        """
        Set hyperparameters.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class DaskRidgeRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, params):
        self.C = params.get('alpha')
        self.model_ = None  # Placeholder for the internal Dask model

    def fit(self, X, y):
        """
        Fit the model using Dask's LinearRegression.
        """
        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        if not isinstance(X, da.Array):
            X = da.from_array(X)
        if not isinstance(y, da.Array):
            y = da.from_array(y)

        self.model_ = LinearRegression(C=self.C)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        X = check_array(X, accept_sparse=True, dtype=None)
        if not isinstance(X, da.Array):
            X = da.from_array(X)
        return self.model_.predict(X).compute()

    def score(self, X, y):
        """
        Returns the accuracy of the model.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        """
        Return hyperparameter dictionary for compatibility with GridSearchCV.
        """
        return {
            "alpha": self.C,
        }

    def set_params(self, **params):
        """
        Set hyperparameters.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class DaskPCA(BaseEstimator, TransformerMixin):
    def __init__(self, params):
        self.n_components = params.get('n_components')
        self.model_ = None

    def fit(self, X):
        """
        Fit the model using Dask's PCA.
        """
        X = check_array(X, accept_sparse=True, dtype=None)
        if not isinstance(X, da.Array):
            X = da.from_array(X)

        self.model_ = PCA(n_components=self.n_components)
        self.model_.fit(X)
        return self

    def transform(self, X):
        """
        Transform the data using the fitted PCA model.
        """
        X = check_array(X, accept_sparse=True, dtype=None)
        if not isinstance(X, da.Array):
            X = da.from_array(X)
        return self.model_.transform(X)

    def get_params(self, deep=True):
        """
        Return hyperparameter dictionary for compatibility with GridSearchCV.
        """
        return {
            "n_components": self.n_components,
        }

    def set_params(self, **params):
        """
        Set hyperparameters.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def inverse_transform(self, X):
        """
        Transform the data back to its original space.
        """
        X = check_array(X, accept_sparse=True, dtype=None)
        if not isinstance(X, da.Array):
            X = da.from_array(X)
        return self.model_.inverse_transform(X)
