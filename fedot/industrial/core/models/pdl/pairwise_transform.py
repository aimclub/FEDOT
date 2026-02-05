import functools
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import sklearn.base
from fedot.core.operations.operation_parameters import OperationParameters
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted


class PDCDataTransformer:
    """
    Transform the data so that it can be processed by PDL models.
    """
    preprocessing_: ColumnTransformer
    preprocessing_y_: ColumnTransformer  # todo fix the ColumnTransformer annotation

    def __init__(self, numeric_features: Iterable = None,
                 ordinal_features: Iterable = None,
                 string_features: Iterable = None,
                 y_type: str = None):
        self.numeric_features = numeric_features
        self.ordinal_features = ordinal_features
        self.string_features = string_features
        if y_type is not None and y_type not in ('numeric', 'ordinal', 'string'):
            raise ValueError(f"y_type must be one of 'numeric', 'ordinal', 'string' but got {y_type}")
        self.y_type = y_type

    def fit(self, X, y=None):

        # y = y.astype('category').cat.codes.astype(np.float32) # todo since I
        # cannot transform the output at least add raise type error on it
        if self.numeric_features is None and self.ordinal_features is None and self.string_features is None:
            self.numeric_features = []
            self.ordinal_features = []  # todo fix name, will be processed a ordinal
            self.string_features = []
            for column in X.columns:
                dtype = X[column].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    self.numeric_features.append(column)
                elif isinstance(dtype, pd.CategoricalDtype):
                    if dtype.ordered:
                        self.ordinal_features.append(column)  # ordinal...
                    else:
                        self.string_features.append(column)
                elif pd.api.types.is_bool_dtype(dtype):  # pd.api.types.is_categorical_dtype(dtype) deprecated
                    self.string_features.append(column)
                elif pd.api.types.is_string_dtype(dtype):
                    self.string_features.append(column)

        X, _ = self.cast_uint(X)
        if self.y_type == 'numeric':
            from sklearn.preprocessing import StandardScaler
            self.preprocessing_y_ = StandardScaler()
        elif self.y_type == 'ordinal':  # string
            from sklearn.preprocessing import OrdinalEncoder
            self.preprocessing_y_ = OrdinalEncoder()
        elif self.y_type == 'string':
            from sklearn.preprocessing import OneHotEncoder
            self.preprocessing_y_ = OneHotEncoder()

        if y is not None and self.preprocessing_y_ is not None:
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y)
            self.preprocessing_y_.fit(y)

        return self

    def cast_uint(self, X: pd.DataFrame, y: pd.Series = None):
        numeric_cols = X.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            X[col] = X[col].astype('float32')

        if y is not None:
            y = y.astype('float32')
        return X, y

    def transform(self, X, y=None):
        check_is_fitted(self)
        X, _ = self.cast_uint(X)
        X = pd.DataFrame(self.preprocessing_.transform(X))
        from scipy.sparse import csr_matrix
        if any(isinstance(e, csr_matrix) for e in X.values.flatten()):
            raise NotImplementedError('error in data \t X contains sparse features (csr_matrix)')
        X = X.dropna(axis=1, how='all')  # Drop columns with all NaN values
        X = X.astype(np.float32)

        if len(X.columns) == 0:
            raise ValueError('error in data \t X no features left after pre-processing')
        # if X.isna().any().any():
        #     raise NotImplementedError('error in data \t Some features are NaNs in the X set')
        if any(x in pd.Series(X.values.flatten()).apply(type).unique() for x in
               ('csr_matrix', 'date',)):  # todo think about adding  'str'
            raise NotImplementedError('error in data \t Dataset contains sparse data')

        if y is not None and self.preprocessing_ is not None:
            y = pd.Series(self.preprocessing_.transform(y), name='y')
        if y is None:
            return X.values
        return X.values, y.values


class SampleWeights:
    def __init__(self, params: Optional[OperationParameters] = None):
        # Save information about the weighting methods as here for better availability
        self.method = params.get('method', 'L2')

        self.method_dict = {
            # Optimization based methods:
            'L2': functools.partial(self._sample_weight_optimize, l2_lambda=0.1),
            'KLD': functools.partial(self._sample_weight_optimize, kld_lambda=0.05),
            'Optimize': self._sample_weight_optimize,
            'L1L2': functools.partial(self._sample_weight_optimize, l1_lambda=0.05, l2_lambda=0.025),
            'L1': functools.partial(self._sample_weight_optimize, l1_lambda=0.1),
            'ExtremeWeightPruning': self._sample_weight_extreme_pruning,
            # Heuristic methods
            'NegativeError': self._sample_weight_negative_error,
            'InverseError': self._sample_weight_inverse_error,
            'OrderedVoting': self._sample_weight_ordered_votes,
            # Other Methods:
            'KMeansClusterCenters': self._sample_weight_by_kmeans_prototypes,
        }

    def _normalize_weights(self, weights: np.ndarray) -> pd.Series:
        """
        Normalize the weights to be between 0 and 1
        :param weights: The weights to be normalized as a pd.Series
        """
        if all(np.isclose(weights, weights.values[0])):
            weights = pd.Series(1., index=weights.index)
        assert weights.min() >= 0, f'Negative weights found: {weights[weights < 0]}'
        weights /= weights.sum()
        return weights

    def __objective_function(self,
                             weights: np.ndarray,
                             pred_val_samples_np: np.ndarray,
                             y_val: np.ndarray,
                             initial_mae: float,
                             kld_lambda=0.,
                             l1_lambda=0.,
                             l2_lambda=0.) -> float:
        assert kld_lambda >= 0, f'kld_lambda should be >=0, got {kld_lambda}'
        assert l1_lambda >= 0, f'l1_lambda should be >=0, got {l1_lambda}'
        assert l2_lambda >= 0, f'l2_lambda should be >=0, got {l2_lambda}'
        assert initial_mae >= 0, f'initial_mae should be >=0, got {initial_mae}'

        predictions = np.matmul(pred_val_samples_np, weights / sum(weights))
        mae = sklearn.metrics.mean_absolute_error(y_val, predictions)

        regularisation = 0
        if kld_lambda > 0:
            train_size = len(weights)
            weights_initial_guess = np.ones(train_size) / train_size
            regularisation += kld_lambda * entropy(weights, weights_initial_guess) / train_size
        if l1_lambda > 0:
            regularisation += l1_lambda * (np.linalg.norm(weights, ord=1) - max(weights))
        if l2_lambda > 0:
            regularisation += l2_lambda * np.linalg.norm(weights, ord=2)

        regularisation *= initial_mae
        loss = mae + regularisation
        return loss

    def _sample_weight_optimize(self, X_val: pd.DataFrame, y_val: pd.Series, kld_lambda=0., l1_lambda=0., l2_lambda=0.,
                                **kwargs) -> pd.Series:
        """
        Minimize the validation MAE using SLSQP optimizer
        with a linear constraint on the sum of the weights.

        :param X_val:
        :param y_val:
        :param kld_lambda: alpha=0.01 i.e. I am ready to loose 1% of the validation MAE to make the solution more general
        :return:
        """
        prediction_samples_df, _ = self._predict_samples(X_val)
        pred_val_samples_np = prediction_samples_df.values
        train_size = len(self.X_train_)
        weights_initial_guess = np.ones(train_size) / train_size
        initial_mae = sklearn.metrics.mean_absolute_error(y_val, np.matmul(pred_val_samples_np, weights_initial_guess))

        def objective_function(weights: np.ndarray) -> float:
            return self.__objective_function(weights=weights, pred_val_samples_np=pred_val_samples_np, y_val=y_val,
                                             initial_mae=initial_mae, kld_lambda=kld_lambda, l1_lambda=l1_lambda,
                                             l2_lambda=l2_lambda)

        variable_bounds = [(0., 1.) for _ in range(train_size)]
        sum_constraint = LinearConstraint(np.ones(train_size), lb=1, ub=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(objective_function, weights_initial_guess, method='SLSQP', bounds=variable_bounds,
                              constraints=[sum_constraint])
        # Extract the solution
        optimal_weight = result.x

        # print("the optimal solution:", optimal_weight)
        # print("Optimal Objective Value, i.e. new log loss validation error:", result.fun)
        sample_weights = pd.Series(optimal_weight, index=self.X_train_.index)
        return sample_weights

    def _sample_weight_extreme_pruning(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        l1 = 0.8
        while l1 > 0.0001:
            weights = self._sample_weight_optimize(X_val=X_val, y_val=y_val, l1_lambda=l1)
            if sum(weights == 0) / len(weights) > .9:
                l1 *= 0.5
            else:
                break
        return weights

    def _error(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        """
        Calculate the Mean Absolute Error for each anchor.
        :param X_val:
        :param y_val:
        :param kwargs:
        :return:
        """
        pred_val_samples, _ = self._predict_samples(X_val)
        errors = pred_val_samples.apply(lambda one_val_samples: abs(y_val - one_val_samples), axis=0)
        val_mae = errors.mean()
        np.testing.assert_array_equal(val_mae.index, self.X_train_.index)
        return val_mae

    def _sample_weight_inverse_error(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        val_mae = self._error(X_val=X_val, y_val=y_val)
        sample_weights = 1. / (val_mae + 0.0001)
        sample_weights = sample_weights / sample_weights.sum()
        return sample_weights

    def _sample_weight_negative_error(self, X_val: pd.DataFrame, y_val: pd.Series, **kwargs) -> pd.Series:
        uniform_weights = pd.Series([1 / len(self.X_train_)] * len(self.X_train_), index=self.X_train_.index)
        val_mae = self._error(X_val=X_val, y_val=y_val)
        if sum(val_mae) == 0:
            return uniform_weights
        sample_weights = ((-val_mae) + max(val_mae)) / sum(val_mae)
        if sum(sample_weights) == 0:
            return uniform_weights
        sample_weights = sample_weights / sample_weights.sum()
        return sample_weights

    @staticmethod
    def _sample_weight_ordered_votes_from_weights(received_weights):
        errors = - received_weights
        k = len(errors)
        ranks = np.argsort(np.argsort(errors)) + 1
        weights = (k - ranks + 1) / (k * (k + 1) / 2)
        return weights

    def _sample_weight_ordered_votes(self, X_val, y_val, force_symmetry=True, **kwargs):
        """
        The best of n anchors gets n votes, the worst gets 1 vote. n is the nb of anchors. Uses the _sample_weight_negative_error function
        for distribution votes.
        works quite good
        :param force_symmetry: Sets the force_symmetry parameter of the prediction function
        :return: The weights as np.NDarray
        """
        weights = self._sample_weight_negative_error(X_val, y_val, force_symmetry=force_symmetry)
        return self._sample_weight_ordered_votes_from_weights(weights)

    def _sample_weight_by_kmeans_prototypes(self, k=None, **kwargs):
        """
        Use KMeans to cluster the train data. Use the k centroids/prototypes found by knn as weights.
        We keep only K anchors that are the prototypes. all other anchors receive a weight of 0

        :param force_symmetry: Sets the force_symmetry parameter of the prediction function
        :param k: The number of prototypes to use. If None, 10% of the training set is used as prototypes
        :return: The weights as np.NDarray
        """
        if not k:
            k = max(int(len(self.X_train_) / 10), 3)  # 10% and min 3 of the training set data points is used as weights

        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        kmeans.fit(self.X_train_)

        cluster_centers = kmeans.cluster_centers_  # Get the cluster centers (prototypical data points)
        distances = cdist(self.X_train_, cluster_centers)  # distance between each data point and each cluster center
        closest_indices = np.argmin(distances, axis=0)  # Get the index of the closest data points to the clusters

        # Create an array to mark the closest data points
        closest_array = np.zeros(len(self.X_train_))
        closest_array[closest_indices] = 1 / k

        s = pd.Series(closest_array, index=self.X_train_.index)
        s = s.fillna(0)  # I don't know why there are NaNs rather than 0s
        assert not s.isna().any(), f'Nans values in sample_weights using KMeans\n {s}'
        return s
