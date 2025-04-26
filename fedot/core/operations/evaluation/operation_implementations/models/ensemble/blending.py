from typing import Optional, Union, Tuple, Callable

import numpy as np
import optuna
from golem.core.log import default_log
from sklearn.metrics import accuracy_score as accuracy, mean_squared_error as mse

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum


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

    def fit(self, input_data: InputData):
        """
        Fits weights for weighted-average blending of model predictions.

        Args:
            input_data: InputData with models predictions

        """
        if self.score_func is None:
            raise ValueError('There is no score function to be optimized. '
                             'Use `regression` or `classification` blending implementations')

        # Constants
        n_preds = input_data.features.shape[1]
        if input_data.task.task_type == TaskTypesEnum.classification:
            n_classes = len(input_data.class_labels)
        else:
            n_classes = None  # there is no classes for regression task
        n_samples = input_data.features.shape[0]
        models = input_data.supplementary_data.previous_operations
        n_models = len(models)

        if self.task_type == 'classification' and n_preds != n_classes * n_models:
            raise ValueError(f"Feature dimensionality mismatch: "
                             f"expected {n_classes * n_models}, got {n_preds}")

        # Weights optimization
        self.log.message(f"Starting weights optimization for models: {models}. "
                         f"Obtained metric - {self.metric.__name__}.")

        # Optuna objective function
        def objective(trial):
            # Suggest weights for each model
            weights = [
                trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                for i in range(n_models)
            ]

            # Normalize weights to sum to 1
            weights = np.array(weights)
            if np.sum(weights) == 0:
                return float('-inf')  # Penalize zero weights

            normalized_weights = weights / np.sum(weights)

            score_args = {
                'weights': normalized_weights,
                'features': input_data.features,
                'n_samples': n_samples,
                'n_models': n_models,
                'target': input_data.target
            }

            if self.task_type == 'classification':
                score_args['n_classes'] = n_classes

            return self.score_func(**score_args)

        # Get optimized weights and score
        optimized_weights, best_score = self._optuna_setup(objective, n_models)

        # Set optimized weights
        model_weight_dict = dict(zip(models, optimized_weights.round(6)))
        self.log.message(f"Optimization result on train set - {self.metric.__name__} = {best_score:.4f}. "
                         f"Models weights: {model_weight_dict}")

        self.weights = optimized_weights
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """
        Get predictions using weighted average blending strategy.

        Args:
            input_data: InputData with models predictions
        Returns:
            OutputData: Aggregated predictions
        """
        # Prepare common arguments
        score_args = {
            'weights': self.weights,
            'features': input_data.features,
            'n_samples': input_data.features.shape[0],
            'n_models': len(input_data.supplementary_data.previous_operations)
        }

        # Add task-specific arguments
        if self.task_type == 'classification':
            score_args['n_classes'] = len(input_data.class_labels)

        # Get predictions based on task type
        result = self.score_func(**score_args)

        # Process result based on task type
        if self.task_type == 'classification':
            labels, _ = result
            output_data = self._convert_to_output(input_data=input_data, predict=labels)
        else:  # regression
            predictions = result
            output_data = self._convert_to_output(input_data=input_data, predict=predictions)

        return output_data

    def _optuna_setup(self, objective: Callable, models_count: int) -> Tuple[np.ndarray, float]:
        """
        Set up and run Optuna optimization for model weights.

        Args:
            objective: Callable function that Optuna will optimize
            models_count: Number of models in the ensemble

        Returns:
            Tuple containing:
                - np.ndarray: Optimized and normalized weights for each model
                - float: Best score achieved during optimization
        """
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
        best_score = study.best_value

        return optimized_weights, best_score


class BlendingClassifier(BlendingImplementation):
    """Implementation of blending for classification tasks"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.metric = accuracy
        self.task_type = 'classification'
        self.score_func = self._get_score

    def predict_proba(self, input_data: InputData) -> OutputData:
        """
        Get probabilities using weighted average blending strategy.

        Args:
            input_data: InputData with models predictions

        Returns:
            OutputData: Probabilities of classes
        """
        _, probs = self.score_func(
            weights=self.weights,
            features=input_data.features,
            n_classes=len(input_data.class_labels),
            n_samples=input_data.features.shape[0],
            n_models=len(input_data.supplementary_data.previous_operations)
        )

        output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return output_data

    def _get_score(
        self, weights: np.ndarray, features: np.ndarray, n_classes: int,
        n_samples: int, n_models: int, target: np.ndarray = None) -> Union[
            float, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate weighted average blending and evaluate its performance.

        Args:
            weights: List of weights for each model
            features: Array of models predictions
            target: True target values
            n_classes: Number of classes in classification task
            n_models: Number of models to blend

        Returns:
            Predicted labels or score
        """
        # Get predictions
        probs = np.zeros((n_samples, n_classes))
        for class_idx in range(n_classes):
            # Get prediction for current class from all models
            class_preds = np.zeros((n_samples, n_models))
            for model_idx in range(n_models):
                col_idx = model_idx * n_classes + class_idx
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

    def _get_score(self, weights: np.ndarray, features: np.ndarray, target: np.ndarray = None, **kwargs) -> Union[
        float, np.ndarray]:
        """
        Calculate weighted average blending and evaluate its performance for regression.

        Args:
            weights: List of weights for each model
            features: Array of models predictions
            target: True target values
            num_samples: Number of samples

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
