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
# from fedot.core.repository.metrics_repository import MetricsRepository, ClassificationMetricsEnum
from sklearn.metrics import accuracy_score as accuracy
from golem.core.log import default_log


class BledingImplementation(ModelImplementation):  # !!
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.max_iter = 50  # !!
        self.seed = 42      # !!
        self.logger = default_log('Blending')
        self.metric = accuracy

    def fit(self, input_data: InputData):
        """ Blending does not provide fit method """
        pass

    def predict(self, input_data: InputData):
        """ Blending does not provide fit method """
        pass


class BlendingClassifier(BledingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

    def predict(self, input_data: InputData) -> OutputData:
        """
        Get prediction using weighted average blending strategy.
        
        Args:
            input_data: InputData with models predictions
            
        Returns:
            OutputData: Labels of blended predictions
        """
        # Add asserts and conditions
        # Constants
        df = pd.read_csv(r"C:\Users\user\Desktop\iris_gbm_stacking_preds.csv").head(44)  # !!!
        features = df.values  # !!!
        target = pd.read_csv(r"C:\Users\user\Desktop\iris_target_03.csv")

        num_classes = len(input_data.class_labels)
        num_samples = features.shape[0]
        models = input_data.supplementary_data.previous_operations
        models_count = len(models)
        task = input_data.task.task_type

        # Accept only models predictions
        assert num_classes * models_count != features.shape[1]

        # Getting optimal weights
        self.logger.info(f"Starting optimization with {models_count} models. Obtained metric - accuracy.")

        def score_func(weights):
            return self._get_score(weights, features, target, num_classes, num_samples, models_count)
        
        optimal_weights = self._optimize(func=score_func, models_count=models_count)

        # Getting predictions and score
        predictions, score = self._get_score(
            optimal_weights, features, target, num_classes, num_samples, models_count, outp_mode=True
        )
        self.logger.info(f"Optimization result - accuracy = {abs(score)}."  # !! hardcode metric
                         f"Models weights: {optimal_weights}")

        # Convert to OutputData and return
        output_data = self._convert_to_output(input_data=input_data, predict=predictions)
        return output_data
        
    def _get_score(self, weights, features, target, num_classes, num_samples, models_count, outp_mode=False):
        """
        Calculate weighted average blending and evaluate its performance.
        
        Args:
            weights: List of weights for each model
            features: Array of models predictions
            target: True target values
            metric: Metric function to evaluate performance
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
        # !!temporal negative score because accuracy as default!!
        score = -self.metric(target, labels)

        if outp_mode:
            return labels, score

        return score

    def _optimize(self, func, models_count):
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


class BlendingRegressor(BledingImplementation):  # !!
    pass


if __name__ == "__main__":
    bld = BlendingClassifier()
    bld.predict(input_data=None)