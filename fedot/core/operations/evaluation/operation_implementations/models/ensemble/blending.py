from typing import Optional, Union, Tuple
from abc import abstractmethod

import numpy as np
import optuna
from golem.core.log import default_log
from sklearn.metrics import accuracy_score as accuracy, mean_squared_error as mse

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class BlendingImplementation(ModelImplementation):
    """Base class for blending operations"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.log = default_log('Blending')

        self.max_iter = 50
        self.seed = 42
        self.metric = None
        self.task_type = None
        self.weights = None
        self.score_func = None
        self.output_mode = 'default'

    def fit(self, input_data: InputData) -> None:
        """
        Fits weights for weighted-average blending of model predictions.

        Args:
            input_data: InputData with models predictions

        """
        if self.score_func is None:
            raise ValueError('There is no score function to be optimized. '
                             'Use `regression` or `classification` blending implementations')

        num_predictions = input_data.features.shape[1]
        num_classes = len(input_data.class_labels)
        num_samples = input_data.features.shape[0]
        models = input_data.supplementary_data.previous_operations
        models_count = len(models)

        if self.task_type == 'classification' and num_predictions != num_classes * models_count:
            raise ValueError(f"Feature dimensionality mismatch: "
                             f"expected {num_classes * models_count}, got {num_predictions}")

        # Weights optimization
        self.log.message(f"Starting weights optimization for models: {models}. "
                         f"Obtained metric - {self.metric.__name__}.")

        def objective(trial):
            # Suggest weights for each model
            weights = [
                trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                for i in range(models_count)
            ]

            # Normalize weights to sum to 1
            weights = np.array(weights)
            if np.sum(weights) == 0:
                return float('-inf')  # Penalize zero weights

            normalized_weights = weights / np.sum(weights)

            return self.score_func(
                weights=normalized_weights,
                features=input_data.features,
                num_classes=num_classes,
                num_samples=num_samples,
                models_count=models_count,
                target=input_data.target
            )

        # Create and run Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=self.max_iter)
        optimized_weights = np.array([
            study.best_params[f'weight_{i}']
            for i in range(models_count)
        ])
        optimized_weights /= np.sum(optimized_weights)

        # Set optimized weights
        model_weight_dict = dict(zip(models, optimized_weights.round(6)))
        self.log.message(f"Optimization result - {self.metric.__name__} = {study.best_value:.4f}. "
                         f"Models weights: {model_weight_dict}")

        self.weights = optimized_weights

    @abstractmethod
    def predict(self, input_data: InputData) -> OutputData:
        """Abstract method. Should be override in child class"""
        raise AbstractMethodNotImplementError


class BlendingClassifier(BlendingImplementation):
    """Implementation of blending for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = accuracy
        self.task_type = 'classification'
        self.score_func = self._get_score

    def predict(self, input_data: InputData) -> OutputData:
        """
        Get labels using weighted average blending strategy.

        Args:
            input_data: InputData with models predictions

        Returns:
            OutputData: Labels of classes
        """
        if self.weights is None:
            raise ValueError('Blending weights are not initialized. Call fit() first.')

        # Get predictions
        labels, _ = self.score_func(
            weights=self.weights,
            features=input_data.features,
            num_classes=len(input_data.class_labels),
            num_samples=input_data.features.shape[0],
            models_count=len(input_data.supplementary_data.previous_operations)
        )
        # Convert to OutputData and return
        output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return output_data

    def predict_proba(self, input_data: InputData) -> OutputData:
        """
        Get probabilities using weighted average blending strategy.

        Args:
            input_data: InputData with models predictions

        Returns:
            OutputData: Probabilities of classes
        """
        if self.weights is None:
            raise ValueError('Blending weights are not initialized. Call fit() first.')

        # Get predictions
        _, probs = self.score_func(
            weights=self.weights,
            features=input_data.features,
            num_classes=len(input_data.class_labels),
            num_samples=input_data.features.shape[0],
            models_count=len(input_data.supplementary_data.previous_operations)
        )
        # Convert to OutputData and return
        output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return output_data

    def _get_score(
        self, weights: np.ndarray, features: np.ndarray, num_classes: int,
        num_samples: int, models_count: int, target: np.ndarray = None) -> Union[
            float, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate weighted average blending and evaluate its performance.

        Args:
            weights: List of weights for each model
            features: Array of models predictions
            target: True target values
            num_classes: Number of classes in classification task
            models_count: Number of models to blend

        Returns:
            Predicted labels or score
        """
        # Get predictions
        probs = np.zeros((num_samples, num_classes))
        for class_idx in range(num_classes):
            # Get prediction for current class from all models
            class_preds = np.zeros((num_samples, models_count))
            for model_idx in range(models_count):
                col_idx = model_idx * num_classes + class_idx
                class_preds[:, model_idx] = features[:, col_idx]

            # Applying weighted average for current class
            probs[:, class_idx] = np.sum(class_preds * weights, axis=1)

        labels = np.argmax(probs, axis=1)

        # If `target` is None, return predicted labels only (inference mode).
        # Otherwise, proceed with optimization (training mode).
        if target is not None:
            score = self.metric(target, labels)
        else:
            return labels, probs

        return score


class BlendingRegressor(BlendingImplementation):
    """Implementation of blending for regression tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = mse
        self.task_type = 'regression'
        self.score_func = self._get_score

    def _get_score(self, weights: np.ndarray, features: np.ndarray,
                   num_samples: int, models_count: int, target: np.ndarray = None) -> Union[
            float, Tuple[np.ndarray, float]]:
        """
        Calculate weighted average blending and evaluate its performance for regression.

        Args:
            weights: List of weights for each model
            features: Array of models predictions
            target: True target values
            num_samples: Number of samples
            models_count: Number of models to blend

        Returns:
            Predicted values or score
        """
        # Get predictions by applying the weights
        predictions = np.dot(features, weights)

        # If `target` is None, return predicted labels only (inference mode).
        # Otherwise, proceed with optimization (training mode).
        if target is not None:
            score = -self.metric(target, predictions)  # minimizing operation for regr
        else:
            return predictions

        return score
