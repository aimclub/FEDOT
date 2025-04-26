from typing import Optional

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class VotingClassifier(ModelImplementation):
    """Hard voting implementation (majority voting)."""

    def __init__(self, params: Optional[OperationParameters] = None, strategy: str = 'hard'):
        super().__init__(params)
        allowed_strategies = ['hard', 'soft']
        if strategy not in allowed_strategies:
            raise ValueError(f"Strategy must be one of {allowed_strategies}, got {strategy}")
        self.strategy = strategy

    def fit(self, input_data: InputData):
        """Voting doesn't provide fit method"""
        return self

    def predict(self, input_data: InputData) -> OutputData:
        """
        Get predictions using voting ensemble strategy.

        Args:
            input_data: InputData with models predictions
        Returns:
            OutputData: Aggregated predictions
        """
        n_classes = len(input_data.class_labels)
        n_models = len(input_data.supplementary_data.previous_operations)
        n_preds = input_data.features.shape[1]

        if n_preds != n_classes * n_models:
            raise ValueError(f"Feature dimensionality mismatch: "
                             f"expected {n_classes * n_models}, got {n_preds}")

        votes = np.zeros(len(input_data.features), dtype=int)
        for i, sample in enumerate(input_data.features):
            # Convert row-based probabilities to per-model prediction matrices
            #
            # Input row format (for 3 models × 3 classes):
            # [cb_p1, cb_p2, cb_p3, xgb_p1, xgb_p2, xgb_p3, lgb_p1, lgb_p2, lgb_p3]
            # Example:
            # [0.7, 0.2, 0.1, 0.6, 0.3, 0.1, 0.6, 0.25, 0.15]
            #
            # Transformed to shape (n_models, n_classes):
            # [
            #   [0.7, 0.2, 0.1],   # CatBoost predictions
            #   [0.6, 0.3, 0.1],    # XGBoost predictions
            #   [0.6, 0.25, 0.15]  # LightGBM predictions
            # ]
            models_probs = sample.reshape(n_models, n_classes)

            final_vote = None
            if self.strategy == 'hard':
                model_votes = np.argmax(models_probs, axis=1)
                final_vote = np.bincount(model_votes).argmax()
            elif self.strategy == 'soft':
                mean_probs = models_probs.mean(axis=0)
                final_vote = mean_probs.argmax()

            votes[i] = final_vote

        output_data = self._convert_to_output(input_data=input_data, predict=votes)
        return output_data
