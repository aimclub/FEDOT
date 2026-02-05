from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
import sklearn.base
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from pymonad.either import Either
from scipy.special import softmax

from fedot.industrial.core.repository.constanst_repository import SKLEARN_CLF_IMP, SKLEARN_REG_IMP


class PairwiseDifferenceEstimator:
    """
    Base class for Pairwise Difference Learning.
    """

    def _convert_to_pandas(self, arr1, arr2):
        if isinstance(arr1, (pd.Series, np.ndarray)):
            arr1 = pd.DataFrame(arr1)
        if isinstance(arr2, (pd.Series, np.ndarray)):
            arr2 = pd.DataFrame(arr2)
        return arr1, arr2

    def _to_pandas_regression(self, *args):
        return (data if data is None or isinstance(data, (pd.DataFrame, pd.Series)) else pd.DataFrame(data) for data in
                args)

    def _pair_data_regression(self, X1, X2, y1=None, y2=None):
        X1, y1, X2, y2 = self._to_pandas_regression(X1, y1, X2, y2)

        X_pair = X1.merge(X2, how="cross")
        x1_pair = X_pair[[f'{column}_x' for column in X1.columns]].rename(
            columns={f'{column}_x': f'{column}_diff' for column in X1.columns})
        x2_pair = X_pair[[f'{column}_y' for column in X1.columns]].rename(
            columns={f'{column}_y': f'{column}_diff' for column in X1.columns})
        X_pair = pd.concat([X_pair, x1_pair - x2_pair], axis='columns')
        # Symmetric
        x2_pair_sym = X_pair[[f'{column}_x' for column in X1.columns]].rename(
            columns={f'{column}_x': f'{column}_y' for column in X1.columns})
        x1_pair_sym = X_pair[[f'{column}_y' for column in X1.columns]].rename(
            columns={f'{column}_y': f'{column}_x' for column in X1.columns})
        X_pair_sym = pd.concat([x1_pair_sym, x2_pair_sym, x2_pair - x1_pair], axis='columns')

        if y1 is not None:
            assert isinstance(y1, pd.Series) or y1.shape[1] == 1, f"Didn't expect more than one output {y1.shape}"
            assert isinstance(y2, pd.Series) or y2.shape[1] == 1, f"Didn't expect more than one output {y2.shape}"

            y_pair = pd.DataFrame(y1).merge(y2, how="cross")
            y_pair_diff = y_pair.iloc[:, 0] - y_pair.iloc[:, 1]
        else:
            y_pair_diff = None

        return X_pair, X_pair_sym, y_pair_diff

    @staticmethod
    def _get_pair_feature_names(features: list) -> list:
        """ Get the new name of features after pairing points. """
        return [f'{name}_x' for name in features] + [f'{name}_y' for name in features]

    def pair_input(self, X1: Union[np.ndarray, pd.Series],
                   X2: Union[np.ndarray, pd.Series]):
        X1, X2 = self._convert_to_pandas(X1, X2)
        X_pair = X1.merge(X2, how="cross")
        x1_pair = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_diff'
                                                                                   for column in X1.columns})
        x2_pair = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_diff'
                                                                                   for column in X1.columns})
        try:
            calculate_difference = x1_pair - x2_pair
        except BaseException:
            raise ValueError(
                "PairwiseDifference: The input data is not compatible with the subtraction operation."
                " Either transform all data to numeric features or use a ColumnTransformer to transform the data.")
        # It means that the input data is not compatible with the subtraction operation.
        # Simply turn all your data into numbers

        X_pair = pd.concat([X_pair, calculate_difference], axis='columns')
        # Symmetric
        x2_pair_sym = X_pair[[f'{column}_x' for column in X1.columns]].rename(columns={f'{column}_x': f'{column}_y'
                                                                                       for column in X1.columns})
        x1_pair_sym = X_pair[[f'{column}_y' for column in X1.columns]].rename(columns={f'{column}_y': f'{column}_x'
                                                                                       for column in X1.columns})
        X_pair_sym = pd.concat([x1_pair_sym, x2_pair_sym, x2_pair - x1_pair], axis='columns')
        # distances = cdist(X1, cluster_centers)
        return X_pair, X_pair_sym

    def pair_output(self,
                    y1: Union[np.ndarray, pd.Series],
                    y2: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """For regresion. beware this is different from regression this is b-a not a-b"""

        y1, y2 = self._convert_to_pandas(y1, y2)
        y_pair = pd.DataFrame(y1).merge(y2, how="cross")
        y_pair_diff = y_pair.iloc[:, 1] - y_pair.iloc[:, 0]
        return y_pair_diff.values

    def pair_output_difference(self,
                               y1: Union[np.ndarray, pd.Series],
                               y2: Union[np.ndarray, pd.Series],
                               nb_classes: int) -> np.ndarray:
        """For MultiClassClassification base on difference only"""
        y1, y2 = self._convert_to_pandas(y1, y2)
        y_pair = pd.DataFrame(y1).merge(y2, how="cross")
        y_pair_diff = (y_pair.iloc[:, 1] != y_pair.iloc[:, 0]).astype(int)
        assert y_pair_diff.nunique() <= 2, f'should only be 0s and 1s {y_pair_diff.unique()}'
        return y_pair_diff.values

    @staticmethod
    def get_pair_feature_names(features: list) -> list:
        """ Get the new name of features after pairing points. """
        return [f'{name}_x' for name in features] + [f'{name}_y' for name in features]

    @staticmethod
    def check_output(y: pd.Series) -> None:
        assert y is not None
        assert isinstance(y, pd.Series)
        assert 'uint' not in str(y.dtype), y.dtype
        assert isinstance(y, pd.Series) or y.shape[1] == 1, f"Didn't expect more than one output {y.shape}"
        assert y.nunique() > 1, y.nunique()
        if y.name is None:
            # just put any name to the output to avoid a bug later
            y.name = 'output'

    @staticmethod
    def check_sample_weight(sample_weight: pd.Series, y_train: pd.Series) -> None:
        if sample_weight is None:
            pass
        elif isinstance(sample_weight, pd.Series):
            # check
            if len(sample_weight) != len(y_train):
                raise ValueError(
                    f'sample_weight size {len(sample_weight)} should be equal to the train size {len(y_train)}')
            if not sample_weight.index.equals(y_train.index):
                raise ValueError(
                    f'sample_weight and y_train must have the same index\n{sample_weight.index}\n{y_train.index}')
            if all(sample_weight.fillna(0) <= 0):
                raise ValueError(f'sample_weight are all negative/Nans.\n{sample_weight}')

            # norm
            class_sums = np.bincount(y_train, sample_weight)
            sample_weight = sample_weight / class_sums[y_train.astype(int)]
        else:
            raise NotImplementedError()

    @staticmethod
    def correct_sample_weight(sample_weight: pd.Series, y_train: pd.Series) -> pd.Series:
        if sample_weight is not None:
            sample_weight = sample_weight / sum(sample_weight)
            # norm
            # class_sums = np.bincount(y_train, sample_weight)
            # sample_weight = sample_weight / class_sums[y_train.astype(int)]

        #     # if sample_weight.min() < 0:  # dolla weight change : improvement +0.0032 bof
        #     #     sample_weight = sample_weight - sample_weight.min()
        return sample_weight

    @staticmethod
    def predict(y_prob: np.ndarray, output_mode: str = 'default', min_label_zero: bool = True):
        if output_mode.__contains__('label'):
            predicted_classes = np.argmax(y_prob, axis=1)[..., np.newaxis]
            predicted_classes = predicted_classes if min_label_zero else predicted_classes + 1
        else:
            predicted_classes = y_prob
        return predicted_classes


class PairwiseDifferenceClassifier:
    """PDL have a low chance of improvement compared to using directly parametric models like Logit, MLP. \
    To obtain an improvement, it is better to use a tree-based model like: ExtraTrees"""

    def __init__(self, params: Optional[OperationParameters] = None):
        self.base_model_params = deepcopy(params._parameters)
        del self.base_model_params['model']
        self.base_model = SKLEARN_CLF_IMP[params.get('model', 'rf')](**self.base_model_params)
        self.pde = PairwiseDifferenceEstimator()
        self.is_model_have_prob_output = hasattr(self.base_model, 'predict_proba')
        self.prior = None
        self.use_prior = False
        self.proba_aggregate_method = 'norm'
        self.sample_weight_ = None

    def _check_target(self):
        if self.target.min() != 0:
            self.target_start_zero = False
        else:
            self.target_start_zero = True

    def _estimate_prior(self):
        if self.prior is not None:
            return self
        # Calculate class priors
        target = pd.DataFrame(self.target)
        class_counts = target.value_counts()
        class_priors = class_counts / len(self.target)
        # Convert class priors to a dictionary
        self.prior = class_priors.sort_index().values

    def fit(self,
            input_data: InputData):
        self.num_classes = input_data.num_classes
        self.target = input_data.target
        self.task_type = input_data.task
        self.is_regression_task = self.task_type.task_type.value == 'regression'
        self.classes_ = sklearn.utils.multiclass.unique_labels(input_data.target)
        self.train_features = input_data.features  # Store the classes seen during fit
        self._estimate_prior()
        self._check_target()
        X_pair, _ = self.pde.pair_input(input_data.features, input_data.features)
        y_pair_diff = self.pde.pair_output_difference(self.target, self.target, self.num_classes)

        self.base_model.fit(X_pair, y_pair_diff)
        return self

    def predict_similarity_samples(self, X: pd.DataFrame, X_anchors=None) -> pd.DataFrame:
        """ For each input sample, output C probabilities for each N train pair.
        Beware that this function does not apply the weights at this level
        """
        if X_anchors is None:
            X_anchors = self.train_features

        X_pair, X_pair_sym = self.pde.pair_input(X, X_anchors)
        if self.is_model_have_prob_output:
            predict_proba = self.base_model.predict_proba
        else:
            def predict_proba(X) -> np.ndarray:
                predictions = self.base_model.predict(X)
                predictions = predictions.astype(int)
                n_samples = len(predictions)
                proba = np.zeros((n_samples, 2), dtype=float)
                proba[range(n_samples), predictions] = 1.
                return proba

        predictions_proba_difference: np.ndarray = predict_proba(X_pair)
        predictions_proba_difference_sym: np.ndarray = predict_proba(X_pair_sym)
        # np.testing.assert_array_equal(predictions_proba_difference.shape, (len(X_pair), 2))
        predictions_proba_similarity_ab = predictions_proba_difference[:, 0]
        predictions_proba_similarity_ba = predictions_proba_difference_sym[:, 0]
        predictions_proba_similarity = (predictions_proba_similarity_ab + predictions_proba_similarity_ba) / 2.

        predictions_proba_similarity_df = pd.DataFrame(
            predictions_proba_similarity.reshape((-1, len(self.train_features))),
            index=pd.DataFrame(X).index, columns=pd.DataFrame(self.train_features).index)
        return predictions_proba_similarity_df

    def __predict_with_prior(self, input_data: np.ndarray, sample_weight):
        tests_trains_classes_likelihood = self.predict_proba_samples(input_data)
        tests_classes_likelihood = self._apply_weights(tests_trains_classes_likelihood, sample_weight)
        np.finfo(tests_classes_likelihood.dtype).eps
        tests_classes_likelihood = tests_classes_likelihood / tests_classes_likelihood.sum(axis=1)[:, np.newaxis]
        tests_classes_likelihood = tests_classes_likelihood.clip(0, 1)
        return tests_classes_likelihood

    def __predict_without_prior(self, input_data: np.ndarray, sample_weight=None):
        X = pd.DataFrame(input_data)
        predictions_proba_similarity_df: pd.DataFrame = pd.DataFrame(self.predict_similarity_samples(X))

        def f(predictions_proba_similarity: pd.Series) -> pd.Series:
            target = pd.Series(self.target.squeeze())
            df = pd.DataFrame(
                {'start': target.reset_index(drop=True), 'similarity': predictions_proba_similarity})
            df = df.fillna(0)
            mean = df.groupby('start', observed=False).mean()['similarity']
            return mean

        tests_classes_likelihood_np = predictions_proba_similarity_df.apply(f, axis='columns')
        # without this normalization it should work for multiclass-multilabel
        if self.proba_aggregate_method == 'norm':
            tests_classes_likelihood_np = tests_classes_likelihood_np.values \
                / tests_classes_likelihood_np.values.sum(axis=-1)[:, np.newaxis]
        elif self.proba_aggregate_method == 'softmax':
            tests_classes_likelihood_np = softmax(tests_classes_likelihood_np, axis=-1)
        return tests_classes_likelihood_np

    def predict_proba_samples(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        # todo add unit test with weight ==[1 1 1 ] and weights = None
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        predictions_proba_similarity: pd.DataFrame = self.predict_similarity_samples(X)

        def g(anchor_class: np.ndarray, predicted_similarity: np.ndarray) -> np.ndarray:
            """

            :param anchor_class: array int
            :param predicted_similarity: array float
            :return:
            """
            prior_cls_probs = (1 - self.prior[anchor_class])
            likelyhood_per_anchor = ((1 - predicted_similarity) / prior_cls_probs)
            likelyhood_per_anchor = likelyhood_per_anchor * self.prior
            n_samples = np.arange(len(likelyhood_per_anchor))
            likelyhood_per_anchor[n_samples, anchor_class] = predicted_similarity
            return likelyhood_per_anchor

        anchor_class = self.target.astype(int)

        def f(predictions_proba_similarity: np.ndarray) -> np.ndarray:
            """ Here we focus on one test point.
            Given its similarity probabilities.
            Return the probability for each class"""
            test_i_trains_classes = g(anchor_class=anchor_class, predicted_similarity=predictions_proba_similarity)
            np.testing.assert_array_equal(test_i_trains_classes.shape, (len(self.target), self.num_classes))
            return test_i_trains_classes

        tests_trains_classes_likelihood = np.apply_along_axis(f, axis=1, arr=predictions_proba_similarity.values)
        return tests_trains_classes_likelihood

    def _apply_weights(self,
                       tests_trains_classes_likelihood: np.ndarray,
                       sample_weight: np.ndarray) -> np.ndarray:
        tests_classes_likelihood = (tests_trains_classes_likelihood *
                                    sample_weight[np.newaxis, :, np.newaxis]).sum(axis=1)
        # np.testing.assert_array_almost_equal(tests_classes_likelihood.sum(axis=-1), 1.)
        return tests_classes_likelihood

    def _abstract_predict(self,
                          input_data: InputData,
                          output_mode: str = 'default'):
        sample_weight = np.full(len(self.target), 1 / len(self.target)) if self.sample_weight_ is None \
            else self.sample_weight_.loc[self.target.index].values

        predict_output = Either(value=input_data.features,
                                monoid=[input_data.features, self.use_prior]).either(
            left_function=lambda features: self.__predict_without_prior(features, sample_weight),
            right_function=lambda features: self.__predict_with_prior(features, sample_weight))
        return self.pde.predict(predict_output, output_mode, self.target_start_zero)

    def predict(self,
                input_data: InputData,
                output_mode: str = 'labels') -> pd.Series:
        """ For each input sample, output one prediction the most probable class.

        """
        return self._abstract_predict(input_data, output_mode)

    def predict_proba(self,
                      input_data: InputData,
                      output_mode: str = 'default') -> pd.Series:
        """ For each input sample, output one prediction the most probable class.

        """

        return self.predict(input_data, output_mode)

    def predict_for_fit(self,
                        input_data: InputData,
                        output_mode: str = 'default'):
        """ For each input sample, output one prediction the most probable class.
        """
        return self.predict(input_data, output_mode)

    def score_difference(self, input_data: InputData) -> float:
        """ WE RETURN THE MAE score XD """
        y_pair_diff = self.pde.pair_output_difference(input_data.target, self.target,
                                                      self.num_classes)  # 0 if similar, 1 if diff
        predictions_proba_similarity: pd.DataFrame = self.predict_similarity_samples(
            input_data.features,
            # reshape=False
        )  # 0% if different, 100% if similar

        return abs(y_pair_diff - (1 - predictions_proba_similarity).values.flatten()).mean()


class PairwiseDifferenceRegressor:
    """PDL have a low chance of improvement compared to using directly parametric models like Ridge, Lasso. \
    To obtain an improvement, it is better to use a tree-based model like: ExtraTrees."""

    def __init__(self, params: Optional[OperationParameters] = None):
        self.base_model_params = deepcopy(params._parameters)
        del self.base_model_params['model']
        self.base_model = SKLEARN_REG_IMP[params.get('model', 'treg')](**self.base_model_params)
        self.pde = PairwiseDifferenceEstimator()
        self.prior = None
        self.use_prior = False
        self.proba_aggregate_method = 'norm'
        self.sample_weight_ = None

    def fit(self,
            input_data: InputData):
        self.num_classes = input_data.num_classes
        self.target = input_data.target
        self.task_type = input_data.task
        self.is_regression_task = self.task_type.task_type.value == 'regression'
        self.train_features = input_data.features  # Store the classes seen during fit
        X_pair, _, y_pair_diff = self.pde._pair_data_regression(self.train_features,
                                                                self.train_features,
                                                                self.target,
                                                                self.target)
        self.base_model.fit(X_pair, y_pair_diff)
        return self

    def predict(self,
                input_data: InputData) -> pd.Series:
        return self._abstract_predict(input_data)

    def predict_proba(self,
                      input_data: InputData) -> pd.Series:
        return self.predict(input_data)

    def predict_for_fit(self,
                        input_data: InputData,
                        output_mode: str = 'default'):
        return self.predict(input_data)

    def _predict_samples(self, input_data: InputData, force_symmetry=True):
        """
        For each input sample, output N predictions (where N = the number of anchors).
        prediction = difference + y_train
        """

        def repeat(s: pd.Series, n_times: int):
            return pd.concat([s] * n_times, ignore_index=True).values

        X = pd.DataFrame(input_data.features)
        final_shape = (-1, len(self.train_features))
        # Create pairs of the new instance each anchor (training instance)
        X_pair, X_pair_sym, _ = self.pde._pair_data_regression(X, self.train_features, None, None)
        # Estimator predicts the difference between each anchor (training instance) and each prediction instance:
        predictions_difference: np.ndarray = self.base_model.predict(X_pair)
        if force_symmetry:
            difference_sym: np.ndarray = self.base_model.predict(X_pair_sym)
            predictions_difference = (predictions_difference - difference_sym) / 2.

        # The known y for the training instances
        predictions_start: np.ndarray = repeat(pd.Series(self.target), n_times=len(X))
        # Combine the difference predicted by the model with the known y => train_y + predicted difference
        predictions: np.ndarray = predictions_start + predictions_difference
        # Set of absolute predictions for each anchor for each prediction instance:
        prediction_samples_df = pd.DataFrame(predictions.reshape(final_shape), index=X.index)
        # The predicted difference to the anchors:
        pred_diff_samples_df = pd.DataFrame(predictions_difference.reshape(final_shape), index=X.index)
        return prediction_samples_df, pred_diff_samples_df

    def __predict_with_weight(self, input_data, prediction_samples_df):
        if isinstance(self.sample_weight_, pd.Series):
            def weighted_avg(samples: pd.Series, weights: pd.Series) -> float:
                weights[weights <= 0] = np.nan
                summed = np.nansum(samples.multiply(weights))
                return summed / np.nansum(weights)

            prediction = prediction_samples_df.apply(
                lambda samples: weighted_avg(samples, self.sample_weight_),
                axis='columns'
            )
        else:
            self.sample_weight_[self.sample_weight_ < 0] = np.nan
            summed = pd.Series(np.nansum(self.sample_weight_, axis=1), index=input_data.index)
            self.sample_weight_ = self.sample_weight_.apply(lambda row: row / summed)
            np.testing.assert_array_almost_equal(self.sample_weight_.sum(axis=1), 1.)
            prediction = (prediction_samples_df * self.sample_weight_).sum(axis=1)
        return prediction

    def _abstract_predict(self, input_data: InputData, force_symmetry=True) -> pd.Series:
        """ For each input sample, output one prediction, the mean of the predicted samples. """
        prediction_samples_df, _ = self._predict_samples(input_data=input_data, force_symmetry=force_symmetry)
        have_weights = isinstance(self.sample_weight_, pd.Series) or isinstance(self.sample_weight_, pd.DataFrame)

        predict_output = Either(value=pd.DataFrame(input_data.features),
                                monoid=[prediction_samples_df, have_weights]).either(
            left_function=lambda features: features.mean(axis=1),
            right_function=lambda init_data: self.__predict_with_weight(init_data, prediction_samples_df))

        return predict_output.values

    def learn_anchor_weights(
            self,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None,
            X_test: pd.DataFrame = None,
            method: str = 'L2',
            enable_warnings=True,
            **kwargs):
        """
        Call this method after the training to create weights for the anchors
        using the given validation data.
        Use the `method` parameter to select one of the following
        weighting methods:
        - 'Optimize': Minimize the validation MAE using the SLSQP optimizer with a linear constraint on the sum of the weights.
        - 'L1': like `Optimize` but includes L1 regularization.
        - 'L2': like `Optimize` but includes L2 regularization.
        - 'L1L2': like `Optimize` but includes L1 and L2 regularization.
        - 'KLD': like `Optimize` but includes a KLD loss to make the weights more uniform.
        - 'ExtremeWeightPruning': lik `L1` but uses  high regularization strength.
        - 'NegativeError': Calculate weights as the negative mean absolute error.
        - 'OrderedVoting': The best of n anchors gets n votes, the worst gets 1 vote. n is the number of anchors.
        - 'KmeansClusterCenters': Calculate weights as the distance to the cluster centers of the KMeans algorithm.
        """
        if y_val is not None:
            old_validation_error = sklearn.metrics.mean_absolute_error(self.predict(X_val), y_val)
        else:
            old_validation_error = 0

        if method not in self._name_to_method_mapping.keys():
            raise NotImplementedError(f"Weighting method {method} unknown! Use one of the following:"
                                      f" '{', '.join(list(self._name_to_method_mapping.keys()))}'")

        sample_weight: pd.Series = self._name_to_method_mapping[method](X_val=X_val, y_val=y_val, X_test=X_test,
                                                                        **kwargs)
        assert not sample_weight.isna().any(), f'Nans values in sample_weights using {method}\n {sample_weight}'
        self.set_sample_weight(sample_weight)
        if y_val is not None:
            new_validation_error = sklearn.metrics.mean_absolute_error(self.predict(X_val), y_val)
            if new_validation_error > old_validation_error and enable_warnings:
                print(f'WARNING: \t new val MAE: {new_validation_error} \t old val MAE:  {old_validation_error}')
        return self

    def set_sample_weight(self, sample_weight: pd.Series):
        """
        Sets the weights for the anchors to the given weights in sample_weight.

        :param sample_weight: The weights for the anchors as a pd.Series
        :return: self (with updated weights)
        """
        if sample_weight is None:
            pass
        elif isinstance(sample_weight, pd.Series):
            if len(sample_weight) != len(self.y_train_):
                raise ValueError(
                    f'sample_weight size {len(sample_weight)} should be equal to the train size {len(self.y_train_)}')
            if not sample_weight.index.equals(self.y_train_.index):
                raise ValueError(
                    f'sample_weight and y_train must have the same index\n{sample_weight.index}\n{self.y_train_.index}')

            if all(sample_weight.fillna(0) == 0):  # All weights are 0 => Set them to 1
                sample_weight = pd.Series(1, index=self.y_train_.index)

            if all(sample_weight.fillna(0) < 0):
                raise ValueError(f'sample_weight are all negative/Nans.\n{sample_weight}')
            if any(pd.isna(sample_weight)):
                raise ValueError(f'sample_weight contains NaNs.\n{sample_weight}')
        else:
            raise ValueError('sample_weight must be a pd.Series')

        self.sample_weight_ = sample_weight
        return self
