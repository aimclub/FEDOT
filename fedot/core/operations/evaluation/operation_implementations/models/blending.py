from typing import Optional

import numpy as np
import optuna
from sklearn.metrics import log_loss, mean_squared_error

from golem.core.log import default_log

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
    ModelImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import TaskTypesEnum


class BlendingImplementation(ModelImplementation):
    """Weighted average blending base class."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.seed = self.params.get('seed', 42)
        self.n_trials = self.params.get('n_trials', 100)
        self.strategy = self.params.get('strategy', 'average')

        self.task = None
        self.classes_ = None
        self.n_classes = None
        self.model_names = None
        self.n_models = None

        self.score_func = None
        self.study = None

        self.weights = None

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.log = default_log('WeightedAverageBlending')

    def _init(self, input_data: InputData):
        """Initialize blending: extract models, setup optuna, and task params."""
        prev_ops = getattr(input_data.supplementary_data, "previous_operations", list())
        self.model_names = prev_ops if prev_ops is not None else []
        self.n_models = len(self.model_names)
        self._init_task_specific_params(input_data)
        if self.n_models >= 2 and self.strategy == 'weighted':
            self.study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                pruner=optuna.pruners.MedianPruner(),
            )

    def _fit(self, input_data: InputData):
        """Fit blending weights (average or optimized)."""
        if self.n_models == 0:
            raise ValueError("No previous models provided for blending.")
        if self.n_models == 1:
            self.weights = np.array([1.0])
            return self
        if self.strategy == 'average':
            self.weights = np.ones(self.n_models) / self.n_models
            return self
        elif self.strategy == 'weighted':
            predictions = self._divide_predictions(input_data)

            def objective(trial):
                weights = np.array([
                    trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                    for i in range(self.n_models)
                ])
                if np.sum(weights) == 0:
                    return float('inf')
                normalized = weights / np.sum(weights)
                blended = self._blend_predictions(predictions, normalized, return_labels=False)
                return self._score(input_data.target, blended)

            self.study.optimize(objective, n_trials=self.n_trials)
            self.weights = np.array([self.study.best_params[f'weight_{i}'] for i in range(self.n_models)])
            self.weights /= np.sum(self.weights)
            return self
        else:
            raise ValueError("Unknown blending strategy. Use 'average' or 'weighted'.")

    def fit(self, input_data: InputData):
        """Initialize and fit blending, log final weights."""
        self._init(input_data)
        self._fit(input_data)

        if self.strategy == 'weighted':
            sorted_pairs = sorted(zip(self.model_names, self.weights), key=lambda x: x[1], reverse=True)
            formula = " + ".join([f"{round(w, 3)} * {model}" for model, w in sorted_pairs])
            self.log.debug(f"Blended prediction = {formula}")
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """Blend predictions using fitted weights."""
        predictions = self._divide_predictions(input_data)
        result = self._blend_predictions(predictions, self.weights, return_labels=True)
        return self._convert_to_output(input_data, result)

    def _init_task_specific_params(self, input_data: InputData):
        """Initialize task-specific parameters."""
        raise NotImplementedError()

    def _score(self, y_true, y_pred):
        """Score function for optimization."""
        raise NotImplementedError()

    def _divide_predictions(self, input_data: InputData):
        """Split predictions by models."""
        raise NotImplementedError()

    def _blend_predictions(self, predictions, weights, return_labels: bool = True):
        """Blend predictions with given weights."""
        raise NotImplementedError()


class BlendingClassifier(BlendingImplementation):
    """Classifier blending (log loss optimized)."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

    def _init_task_specific_params(self, input_data: InputData):
        self.task = TaskTypesEnum.classification
        self.classes_ = input_data.class_labels
        self.n_classes = input_data.num_classes

    def _score(self, y_true, y_pred):
        return log_loss(y_true, y_pred)

    def _blend_predictions(self, predictions, weights, return_labels=True):
        blended = np.average(predictions, axis=0, weights=weights)
        if return_labels:
            if self.n_classes > 2:
                return np.argmax(blended, axis=1)
            else:
                return (blended > 0.5).astype(int)
        return blended

    def _divide_predictions(self, input_data: InputData):
        predictions = input_data.features
        split_size = 1 if self.n_classes == 2 else self.n_classes
        expected_features = self.n_models * split_size

        if predictions.shape[1] != expected_features:
            problem_type = "binary" if self.n_classes == 2 else "multiclass"
            raise ValueError(f"Shape mismatch for {problem_type} classification")

        return np.split(predictions, self.n_models, axis=1)

    def predict_proba(self, input_data: InputData) -> OutputData:
        predictions = self._divide_predictions(input_data)
        result = self._blend_predictions(predictions, self.weights, return_labels=False)
        return self._convert_to_output(input_data, result)


class BlendingRegressor(BlendingImplementation):
    """Regressor blending (MSE optimized)."""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

    def _init_task_specific_params(self, input_data: InputData):
        self.task = TaskTypesEnum.regression

    def _score(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def _blend_predictions(self, predictions, weights, return_labels=True):
        return np.average(predictions, axis=0, weights=weights)

    def _divide_predictions(self, input_data: InputData):
        predictions = input_data.features
        if predictions.shape[1] != self.n_models:
            raise ValueError("Shape mismatch for regression")
        return np.split(predictions, self.n_models, axis=1)
