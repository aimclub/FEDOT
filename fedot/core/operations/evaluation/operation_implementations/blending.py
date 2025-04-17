from typing import Callable, Optional, Union, Tuple

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
# from fedot.core.repository.metrics_repository import MetricsRepository, ClassificationMetricsEnum
from sklearn.metrics import accuracy_score as accuracy, root_mean_squared_log_error as rmse
from golem.core.log import default_log


class BlendingImplementation(ModelImplementation):
    """Base class for blending operations"""
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.max_iter = 50  # !!
        self.seed = 42      # !!
        self.logger = default_log('Blending')
        self.metric = None  # !!
        self.task_type = None

    def fit(self, input_data: InputData):
        """
        Blending does not provide fit method
        """
        pass

    def predict(self, input_data: InputData):
        """
        Abstract method to be implemented in child classes
        """
        raise NotImplementedError("Implement in child class")

    def _optimize(self, func: Callable, models_count: int):
        """
        Perform Bayesian optimization to find optimal weights.

        Args:
            func: Scoring function that accepts weights
            models_count: Number of models to optimize weights for

        Returns:
            Array of optimized weights
        """
        search_space = [Real(0.0, 1.0, name=f'weight_{i}') for i in range(models_count)]

        result = gp_minimize(
            func,
            search_space,
            n_calls=self.max_iter,
            random_state=self.seed,
            verbose=False
        )

        # Return normalized weights
        optimal_weights = np.array(result.x) / np.sum(result.x)
        return optimal_weights


class BlendingClassifier(BlendingImplementation):
    """Implementation of blending for classification tasks"""
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = accuracy
        self.task_type = 'classification'

    def predict(self, input_data: InputData) -> OutputData:
        """
        Get prediction using weighted average blending strategy.
        
        Args:
            input_data: InputData with models predictions
            
        Returns:
            OutputData: Labels of blended predictions
        """
        features = input_data.features
        target = input_data.target

        num_classes = len(input_data.class_labels)
        num_samples = features.shape[0]
        models = input_data.supplementary_data.previous_operations
        models_count = len(models)

        if features.shape[1] != num_classes * models_count:
            self.logger.warning(
                f"Feature dimensionality mismatch: expected {num_classes * models_count}, got {features.shape[1]}")

        # Weights optimization
        self.logger.info(f"Starting optimization with {models_count} models: {models}. Obtained metric - !!!.")

        def score_func(weights):
            return self._get_score(weights, features, target, num_classes, num_samples, models_count)
        
        optimal_weights = self._optimize(func=score_func, models_count=models_count)

        # Get predictions and score
        predictions, score = self._get_score(
            optimal_weights, features, target, num_classes, num_samples, models_count, outp_mode=True
        )
        model_weight_dict = dict(zip(models, optimal_weights))
        self.logger.info(f"Optimization result - accuracy = {abs(score)}."  # !! hardcode metric
                         f"Models weights: {model_weight_dict}")

        # Convert to OutputData and return
        output_data = self._convert_to_output(input_data=input_data, predict=predictions)
        return output_data
        
    def _get_score(
            self, weights: np.ndarray, features: np.ndarray, target: np.ndarray,
            num_classes: int, num_samples: int, models_count: int, outp_mode=False) -> Union[
        float, Tuple[np.ndarray, float]]:
        """
        Calculate weighted average blending and evaluate its performance.

        Args:
            weights: List of weights for each model
            features: Array of models predictions
            target: True target values
            num_classes: Number of classes in classification task
            models_count: Number of models to blend
            outp_mode: Switch to output mode from optimization mode
            
        Returns:
            Predicted labels or(and) score
        """
        # Weights normalization
        weights = np.array(weights) / np.sum(weights)

        # Get predictions
        result = np.zeros((num_samples, num_classes))
        for class_idx in range(num_classes):
            # Get prediction for current class from all models
            class_preds = np.zeros((num_samples, num_classes))
            for model_idx in range(models_count):
                col_idx = model_idx * num_classes + class_idx
                class_preds[:, model_idx] = features[:, col_idx]

            # Applying weighted average for current class
            result[:, class_idx] = np.sum(class_preds * weights, axis=1)

        # Result normalization
        row_sums = result.sum(axis=1, keepdims=True)
        normalized_result = result / row_sums

        labels = np.argmax(normalized_result, axis=1)
        # Because gp_minimize is minimizing operation !!! look at metric
        score = -self.metric(target, labels)

        if outp_mode:
            return labels, score

        return score


class BlendingRegressor(BlendingImplementation):
    """Implementation of blending for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = rmse
        self.task_type = 'regression'

    def predict(self, input_data: InputData) -> OutputData:
        """
        Get prediction using weighted average blending strategy for regression.

        Args:
            input_data: InputData with models predictions

        Returns:
            OutputData: Blended regression predictions
        """
        features = input_data.features
        target = input_data.target

        num_samples = features.shape[0]
        models = input_data.supplementary_data.previous_operations
        models_count = len(models)

        # Weights optimization
        self.logger.info(f"Starting optimization with {models_count} models: {models}. Obtained metric - !!!.")

        def score_func(weights):
            return self._get_score(weights, features, target, num_samples, models_count)

        optimal_weights = self._optimize(func=score_func, models_count=models_count)

        # Get predictions and score
        predictions, score = self._get_score(
            optimal_weights, features, target, num_samples, models_count, outp_mode=True
        )
        model_weight_dict = dict(zip(models, optimal_weights))
        self.logger.info(f"Optimization result - accuracy = {abs(score)}."  # !! hardcode metric
                         f"Models weights: {model_weight_dict}")

        # Convert to OutputData and return
        output_data = self._convert_to_output(input_data=input_data, predict=predictions)
        return output_data

    def _get_score(self, weights: np.ndarray, features: np.ndarray, target: np.ndarray,
                   num_samples: int, models_count: int, outp_mode: bool = False) -> Union[
        float, Tuple[np.ndarray, float]]:
        """
        Calculate weighted average blending and evaluate its performance for regression.

        Args:
            weights: List of weights for each model
            features: Array of models predictions
            target: True target values
            num_samples: Number of samples
            models_count: Number of models to blend
            outp_mode: Switch to output mode from optimization mode

        Returns:
            Predicted values or(and) score
        """
        # Weights normalization
        weights = np.array(weights) / np.sum(weights)

        # Get predictions by applying the weights
        predictions = np.zeros(num_samples)
        for model_idx in range(models_count):
            predictions += weights[model_idx] * features[:, model_idx]

        # Because gp_minimize is minimizing operation !!! look at metric
        score = -self.metric(target, predictions)

        if outp_mode:
            return predictions, score

        return score
