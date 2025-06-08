from typing import Optional

import numpy as np
import optuna
from sklearn.metrics import log_loss, mean_squared_error

from golem.core.log import default_log

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum


class BlendingImplementation(ModelImplementation):
    """Base class for weighted average blending."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.task = None
        self.n_trials = 100
        self.classes_ = None
        self.n_classes = None
        self.models = None
        self.n_models = None

        self.score_func = None
        self.study = None

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.log = default_log('WeightedAverageBlending')

    def _init(self, input_data: InputData):
        self.task = input_data.task.task_type
        self.classes_ = input_data.class_labels
        self.n_classes = input_data.num_classes
        self.models = input_data.supplementary_data.previous_operations
        self.n_models = len(self.models)
        self.score_func = self._setup_default_score_func()
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

    def _fit(self, input_data: InputData):
        """Method for weights optimization."""
        if self.n_models == 0:
            raise ValueError("No previous models provided for blending.")
        if self.n_models == 1:
            self.log.message(f"Got only one model; using weight 1.0 for {self.models[0]}")
            self.weights = np.array([1.0])
            return self

        predictions = self._divide_predictions(input_data=input_data)

        def objective(trial):
            # Suggest weights for each model
            weights = [
                trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                for i in range(self.n_models)
            ]

            # Normalize weights to sum to 1
            weights = np.array(weights)
            if np.sum(weights) == 0:
                return float('inf')  # Penalize zero weights
            normalized_weights = weights / np.sum(weights)

            blended_pred = self._blend_predictions(predictions, normalized_weights, return_labels=False)
            return self.score_func(input_data.target, blended_pred)

        self.log.message(f"Starting weights optimization for models: {self.models}. "
                    f"Obtained score function - {self.score_func.__name__}.")

        self.study.optimize(objective, n_trials=self.n_trials)

        self.log.message(f"Optimization completed. Best {self.score_func.__name__} score: {self.study.best_value:.6f}")

        optimized_weights = np.array([
            self.study.best_params[f'weight_{i}']
            for i in range(self.n_models)
        ])
        optimized_weights /= np.sum(optimized_weights)

        self.weights = optimized_weights
        return self

    def fit(self, input_data: InputData):
        """
        Fit weights for weighted-average blending of model predictions.

        Args:
            input_data: InputData with models predictions
        """
        self._init(input_data)

        # Get weights
        self._fit(input_data=input_data)

        # Logging weights result
        sorted_pairs = sorted(zip(self.models, self.weights), key=lambda x: x[1], reverse=True)
        weight_formula = " + ".join([f"{round(w, 3)} * {model}" for model, w in sorted_pairs])
        self.log.message(f"Blended prediction = {weight_formula}")
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """
        Get predictions using weighted average blending strategy.

        Args:
            input_data: InputData with models predictions
        """
        models_predictions = self._divide_predictions(input_data)
        result = self._blend_predictions(models_predictions, self.weights, return_labels=True)
        return self._convert_to_output(input_data=input_data, predict=result)

    def predict_proba(self, input_data: InputData) -> OutputData:
        """
        Predict class probabilities using weighted average blending strategy.

        Args:
            input_data: InputData with models predictions
        """
        if not self.task == TaskTypesEnum.classification:
            raise ValueError("predict_proba is only available for classification tasks.")
        models_predictions = self._divide_predictions(input_data)
        result = self._blend_predictions(models_predictions, self.weights, return_labels=False)
        return self._convert_to_output(input_data=input_data, predict=result)

    def _blend_predictions(self, predictions: list[np.ndarray], weights: np.ndarray, return_labels: bool = True):
        """Blend predictions using weighted average."""
        blended = np.average(predictions, axis=0, weights=weights)

        if self.task == TaskTypesEnum.classification and return_labels:
            if self.n_classes is not None:
                if self.n_classes > 2:
                    return np.argmax(blended, axis=1)
                else:
                    return (blended > 0.5).astype(int)
            else:
                raise ValueError("Got None in number of classes.")
        else:
            return blended

    def _setup_default_score_func(self):
        """Setup default score function to be optimized."""
        if self.task == TaskTypesEnum.classification:
            return log_loss
        elif self.task == TaskTypesEnum.regression or self.task == TaskTypesEnum.ts_forecasting:
            return mean_squared_error
        else:
            raise ValueError(f"Can't get score function for the task {self.task}.")

    def _divide_predictions(self, input_data: InputData) -> list:
        """Split concatenated predictions from different models into separate arrays."""
        predictions = input_data.features
        preds_list = []

        if self.task == TaskTypesEnum.classification:
            if self.n_classes == 2:
                # Binary classification case - one probability per model
                if predictions.shape[1] != self.n_models:
                    raise ValueError(f"For binary classification expected {self.n_models} columns, "
                                f"got {predictions.shape[1]}")
                # Split into separate model predictions
                for i in range(self.n_models):
                    model_pred = predictions[:, i:i+1]  # Keep as 2D array
                    preds_list.append(model_pred)
            else:
                # Multiclass case - n_classes probabilities per model
                if predictions.shape[1] != self.n_classes * self.n_models:
                    raise ValueError(f"For multiclass classification expected {self.n_classes * self.n_models} columns, "
                                f"got {predictions.shape[1]}")
                # Split into separate model predictions
                for i in range(self.n_models):
                    start = i * self.n_classes
                    end = (i + 1) * self.n_classes
                    model_pred = predictions[:, start:end]
                    preds_list.append(model_pred)
        else:
            # Regression or time series forecasting case - one value per model
            if predictions.shape[1] != self.n_models:
                raise ValueError(f"For regression expected {self.n_models} columns, "
                            f"got {predictions.shape[1]}")
            # Split into separate model predictions
            for i in range(self.n_models):
                model_pred = predictions[:, i:i+1]  # Keep as 2D array
                preds_list.append(model_pred)

        return preds_list
