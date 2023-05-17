from abc import ABC, abstractmethod
import copy
from typing import Optional

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters, get_default_params


class BaseKFoldBagging:
    @abstractmethod
    def __init__(self, n_layers: int, n_repeated: int, k_fold: int):
        self.n_layers = n_layers
        self.n_repeated = n_repeated
        self.k_fold = k_fold

    def _splitting_data_into_chunks(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Get indices for train_fold and test_fold data """
        sgkf = StratifiedKFold(n_splits=self.k_fold)
        chunks = []

        for train_indices, test_indices in sgkf.split(X=X, y=y.reshape(-1)):
            chunks.append([train_indices.tolist(), test_indices.tolist()])

        return chunks

    def _get_avg_oof_models_preds(self, oof_probs: np.ndarray) -> np.ndarray:
        avg_off_models_pred = []

        for model_index in range(self.k_fold):
            sum_per_chunk = np.sum(oof_probs[:, :, model_index, :, :], axis=0)  # Sum model's Y_hat in fold
            probs_per_chunk = sum_per_chunk / self.n_repeated  # Average OOF predictions model's
            y_hat_m = np.argmax(probs_per_chunk, axis=2).reshape(-1)  # Get labels from prediction

            avg_off_models_pred.append(y_hat_m)

        return np.array(avg_off_models_pred)

    def _concatenate_prev_data_and_preds(self, X: np.ndarray, oof_preds: np.ndarray) -> np.ndarray:
        """ Concatenate predicted data for fitting new layers """
        return np.concatenate((X, oof_preds.T), axis=1)

    def _get_by_indices(self, data, indices):
        return np.take(data[0].T, indices, axis=-1).T, np.take(data[1], indices)


class KFoldBaggingClassifier(BaseKFoldBagging):
    def __init__(self, base_estimator: object, n_layers: int, n_repeated: int, k_fold: int):
        super().__init__(
            n_layers=n_layers,
            n_repeated=n_repeated,
            k_fold=k_fold
        )

        self.n_layers = n_layers
        self.n_repeated = n_repeated
        self.k_fold = k_fold

        self.base_estimator = base_estimator
        self.ensemble_layers = [
            [copy.deepcopy(base_estimator) for _ in range(self.k_fold)] for _ in range(self.n_layers)
        ]

        self._decision_function = self.average_voting

    def average_voting(self, models_prediction):
        predictions = np.argmax(np.sum(models_prediction, axis=0) / self.k_fold, axis=-1)

        return predictions

    # TODO: Change InputData to X, y (np.ndarray, np.ndarray)
    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Method to train chosen operation with provided data

        Args:
            X: train data
            y: target data

        Returns:
            trained bagging model
        """

        for layer in range(self.n_layers):
            # Multi-layer stack ensembling
            # out_of_fold_probs.shape - (n_repeats, models, n_fold, test_cols, probs)
            out_of_fold_probs = []

            for i in range(self.n_repeated):
                # Each model in layer fits n-repeated times at folds
                chunks_preds = []

                # Randomly split data into k disjoint chunks
                chunks = self._splitting_data_into_chunks(X=X, y=y)

                for j in range(self.k_fold):
                    # Each model trained at training data (X^{-j}, y^{-j})
                    # and make prediction for fold data (X^{j}, y^{j})
                    # models_preds.shape - (n_fold, test_cols, probs)
                    models_memory_id = []
                    models_preds = []

                    # TODO: Parallel models fitting
                    for model in self.ensemble_layers[layer]:
                        # Get training
                        X_train_subset, y_train_subset = self._get_by_indices((X, y), chunks[j][0])
                        X_test_subset, _ = self._get_by_indices((X, y), chunks[j][1])

                        model.fit(X=X_train_subset, y=y_train_subset.reshape(-1, 1))
                        models_memory_id.append(hex(id(model)))
                        models_preds.append(model.predict_proba(X_test_subset))

                    chunks_preds.append(models_preds)
                out_of_fold_probs.append(chunks_preds)

            avg_oof_models_preds = self._get_avg_oof_models_preds(np.array(out_of_fold_probs))
            X = self._concatenate_prev_data_and_preds(X, avg_oof_models_preds)

        return self.ensemble_layers

    def predict(self, X: np.ndarray) -> np.ndarray:
        for layer in range(self.n_layers):
            # Shape - ()
            models_probs = []

            for model_index, model in enumerate(self.ensemble_layers[layer]):
                models_probs.append(model.predict_proba(X))

            # Check if layer is last
            if layer != self.n_layers - 1:
                # Get labels from model's preds probs
                models_preds = np.argmax(np.array(models_probs), axis=-1)

                # Adding new features for next layer
                X = self._concatenate_prev_data_and_preds(X, models_preds)

            else:
                # Get preds from models in last layers
                prediction = self._decision_function(models_probs)

        return prediction

    def predict_proba(self, predict_data: InputData) -> OutputData:
        NotImplementedError()


# class KFoldBaggingRegression(BaseKFoldBagging):