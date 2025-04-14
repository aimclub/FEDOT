from typing import Callable, Optional

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
from fedot.core.repository.metrics_repository import MetricsRepository


class BlendingImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.max_iter = 50

    def fit(self, input_data: InputData):
        """ Blending does not provide fit method """
        pass

    def predict(self, input_data: InputData) -> OutputData:
        """ Get prediction with chosen strategy. Weighted average set as default

        :param input_data: metadata - models predictions
        """
        metric = MetricsRepository.get_metric()  # !!!!

        df = pd.read_csv(r"C:\Users\user\Desktop\iris_gbm_stacking_preds.csv")
        array = df.values

        num_classes = 3  # !!!!
        num_samples = array.shape[0]
        models_count = array.shape[1] // num_classes

        # Equals weights initialization
        weights = [1 / num_classes] * num_classes

        search_space = [Real(0.0, 1.0, name=f'weight_{i}') for i in range(models_count)]

        result = gp_minimize(
            self._get_score,
            search_space,
            n_calls=self.max_iter,
            random_state=42,
            verbose=True
        )

        optimal_weights = np.array(result.x) / np.sum(result.x)

    def _get_score(self, weights, y_pred, y_true, metric, num_samples, num_classes, models_count):
        # Get predictions
        result = np.zeros((num_samples, num_classes))
        for class_idx in range(num_classes):
            # Get prediction for current class from all models
            class_preds = np.zeros((num_samples, num_classes))
            for model_idx in range(models_count):
                col_idx = model_idx * num_classes + class_idx
                class_preds[:, model_idx] = y_pred[:, col_idx]

            # Applying weighted average for current class
            result[:, class_idx] = np.sum(class_preds * weights, axis=1)

        # Normalization to get probabilities
        row_sums = result.sum(axis=1, keepdims=True)
        normalized_result = result / row_sums

        # !! протестировать это
        labels = np.argmax(normalized_result, axis=1)
        score = metric(y_true, labels)

        return score


if __name__ == "__main__":
    bld = BlendingImplementation()
    bld.predict(input_data=None)