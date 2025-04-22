from typing import Callable, Optional, Union, Tuple

import numpy as np
from golem.core.log import default_log
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import accuracy_score as accuracy, mean_squared_error as mse

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class BlendingImplementation(ModelImplementation):
    """Base class for blending operations"""
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.max_iter = 50
        self.seed = 42
        self.metric = None
        self.task_type = None
        self.weights = None
        self.score_func = None
        self.log = default_log('Blending')

    def fit(self, input_data: InputData):
        """
        Fitting weights for weight-average blending strategy.

        Args:
            input_data: InputData with models predictions

        """
        num_predictions = input_data.features.shape[1]
        num_classes = len(input_data.class_labels)
        num_samples = input_data.features.shape[0]
        models = input_data.supplementary_data.previous_operations
        models_count = len(models)

        if self.task_type == 'classification' and num_predictions != num_classes * models_count:
            self.log.warning(
                f"Feature dimensionality mismatch: expected {num_classes * models_count}, got {num_predictions}")

        # Weights optimization
        self.log.message(f"Starting optimization with models: {models}. "
                      f"Obtained metric - {self.metric.__name__}.")

        def score_func(weights):
            return self.score_func(
                weights=weights,
                features=input_data.features,
                num_classes=num_classes,
                num_samples=num_samples,
                models_count=models_count,
                target=input_data.target
            )

        optimized_weights = self._optimize(func=score_func, models_count=models_count).round(6)

        # Set optimized weights
        score = score_func(optimized_weights)
        model_weight_dict = dict(zip(models, optimized_weights))
        self.log.message(f"Optimization result - {self.metric.__name__} = {abs(score)}. "
                      f"Models weights: {model_weight_dict}")

        self.weights = optimized_weights

    def predict(self, input_data: InputData):
        """
        Get prediction using weighted average blending strategy.

        Args:
            input_data: InputData with models predictions

        Returns:
            OutputData: Blended predictions
        """
        labels = self.score_func(
            weights=self.weights,
            features=input_data.features,
            num_classes=len(input_data.class_labels),
            num_samples=input_data.features.shape[0],
            models_count=len(input_data.supplementary_data.previous_operations)
        )
        # Convert to OutputData and return
        output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return output_data

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
        optimized_weights = np.array(result.x) / np.sum(result.x)
        return optimized_weights


class BlendingClassifier(BlendingImplementation):
    """Implementation of blending for classification tasks"""
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = accuracy
        self.task_type = 'classification'
        self.score_func = self._get_score

    def _get_score(
            self, weights: np.ndarray, features: np.ndarray, num_classes: int,
            num_samples: int, models_count: int, target: np.ndarray=None) -> Union[
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
        probs = result / row_sums

        labels = np.argmax(probs, axis=1)

        # If `target` is None, return predicted labels only (inference mode).
        # Otherwise, proceed with optimization (training mode).
        if target is not None:
            score = -self.metric(target, labels)  # gp_minimize is minimizing operation
        else:
            return labels

        return score


class BlendingRegressor(BlendingImplementation):
    """Implementation of blending for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = mse
        self.task_type = 'regression'
        self.score_func = self._get_score

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

        # Because gp_minimize is minimizing operation
        score = -self.metric(target, predictions)

        if outp_mode:
            return predictions, score

        return score
