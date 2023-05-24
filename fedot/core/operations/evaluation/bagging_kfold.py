from abc import ABC, abstractmethod
import copy
from typing import Optional

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score as roc_auc

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters, get_default_params


class BaseKFoldBagging:
    @abstractmethod
    def __init__(self, n_layers: int, n_repeats: int, k_fold: int):
        self.n_layers = n_layers
        self.n_repeated = n_repeats
        self.k_fold = k_fold

        self._cv_splitter = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats)

    def _splitting_data_into_chunks(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return next(self._cv_splitter.split(X=X, y=y.reshape(-1)))

    def _get_avg_oof_models_preds(self, oof_probs: np.ndarray, oof_repeats: np.ndarray) -> np.ndarray:
        oof_repeats_without_0 = np.where(oof_repeats == 0, 1, oof_repeats)
        layer_probs = oof_probs / oof_repeats_without_0.reshape(-1, 1)
        layer_preds = np.argmax(layer_probs, axis=1).reshape(-1, 1)

        return layer_preds

    def _concatenate_prev_data_and_preds(self, X: np.ndarray, oof_preds: np.ndarray) -> np.ndarray:
        """ Concatenate predicted data for fitting new layers """
        return np.concatenate((X, oof_preds), axis=1)

    def _get_by_indices(self, data, indices):
        return np.take(data[0].T, indices, axis=-1).T, np.take(data[1], indices)


class KFoldBaggingClassifier(BaseKFoldBagging):
    def __init__(self, base_estimator: object, n_layers: int, n_repeats: int, k_fold: int):
        super().__init__(
            n_layers=n_layers,
            n_repeats=n_repeats,
            k_fold=k_fold
        )

        self.n_layers = n_layers
        self.n_repeats = n_repeats
        self.k_fold = k_fold

        self.base_estimator = base_estimator
        self.ensemble_layers = [
            [copy.deepcopy(base_estimator) for _ in range(self.k_fold)] for _ in range(self.n_layers)
        ]

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
            # Multi-layer stack ensembeling
            # out_of_fold_probs.shape - (n_repeats, models, test_cols, probs)

            out_of_fold_probs = np.zeros(shape=(y.shape[0], 2))
            out_of_fold_repeats = np.zeros(shape=y.shape)

            for i in range(self.n_repeats):
                # Each model in layer fits n-repeated times at folds
                # Randomly split data into k disjoint chunks

                for j in range(self.k_fold):
                    # Each model trained at training data (X^{-j}, y^{-j})
                    # and make prediction for fold data (X^{j}, y^{j})

                    # TODO: Parallel models fitting
                    for id, model in enumerate(self.ensemble_layers[layer]):
                        train_indices, test_indices = self._splitting_data_into_chunks(X=X, y=y)
                        X_train_subset, y_train_subset = self._get_by_indices((X, y), train_indices)
                        X_val_subset, y_val_subset = self._get_by_indices((X, y), test_indices)

                        model.fit(X=X_train_subset, y=y_train_subset.reshape(-1, 1))

                        fit_proba = model.predict_proba(X_train_subset)
                        fit_labels = np.argmax(fit_proba, axis=1)

                        pred_proba = model.predict_proba(X_val_subset)
                        pred_labels = np.argmax(pred_proba, axis=1)

                        # TODO: val_score from metric of fedot
                        fit_score = roc_auc(y_true=y_train_subset, y_score=fit_labels)
                        val_score = roc_auc(y_true=y_val_subset, y_score=pred_labels)

                        # TODO: Fix
                        out_of_fold_probs[test_indices] += pred_proba
                        out_of_fold_repeats[test_indices] += 1

            avg_oof_models_preds = self._get_avg_oof_models_preds(out_of_fold_probs, out_of_fold_repeats)

            X = self._concatenate_prev_data_and_preds(X, avg_oof_models_preds)

        return self.ensemble_layers

    def predict(self, X: np.ndarray) -> np.ndarray:
        for layer in range(self.n_layers):
            out_of_fold_probs = np.zeros(shape=(X.shape[0], 2))

            for model_index, model in enumerate(self.ensemble_layers[layer]):
                model_probs = model.predict_proba(X)
                out_of_fold_probs += model_probs

            # Check if layer is last
            if layer != self.n_layers - 1:
                models_probs = out_of_fold_probs / self.k_fold
                models_preds = np.argmax(models_probs, axis=1).reshape(-1, 1)

                # Adding new features for next layer
                X = self._concatenate_prev_data_and_preds(X, models_preds)

            else:
                # Get preds from models in last layers
                prediction = models_preds

        return prediction

    def predict_proba(self, predict_data: InputData) -> OutputData:
        NotImplementedError()


# class KFoldBaggingRegression(BaseKFoldBagging):