import copy
import time
from abc import abstractmethod

import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from fedot.core.operations.evaluation.operation_implementations.models.bag_ensembles.fold_fitting_strategy import \
    SequentialFoldFittingStrategy, ParallelFoldFittingStrategy


class BaseKFoldBagging:
    @abstractmethod
    def __init__(self,
                 model_base,
                 n_repeats: int = 0,
                 k_fold: int = 5,
                 fold_fitting_strategy: str = None,
                 n_jobs: int = 0,
                 ):
        self.model_base = model_base
        # Store special
        self.models = []

        self.n_repeated = n_repeats
        self.k_fold = k_fold

        self._oof_pred_proba = None
        self._oof_pred_model_repeats = None

        self.X_new = None
        self.unique_class = None
        self.cv_splitter = self._get_cv_splitter(n_repeats, k_fold)
        self.kfold_fitting_strategy = self._get_fold_fitting_stratgey(fold_fitting_strategy)

    @staticmethod
    def _get_fold_fitting_stratgey(fold_fitting_strategy):
        if fold_fitting_strategy == 'sequential':
            fold_fitting_strategy = SequentialFoldFittingStrategy

        # TODO: parallel for cpu and gpu
        elif fold_fitting_strategy == 'parallel':
            fold_fitting_strategy = ParallelFoldFittingStrategy

        else:
            raise ValueError(f'{fold_fitting_strategy} is unknown fitting strategy')

        return fold_fitting_strategy

    def _get_fold_fitting_args(self, X: np.ndarray, y: np.ndarray, oof_pred_proba: np.ndarray,
                               oof_pred_model_repeats: np.ndarray):
        return dict(
            model_base=self.model_base,
            bagged_ensemble_model=self,
            X=X, y=y,
            oof_pred_proba=oof_pred_proba,
            oof_pred_model_repeats=oof_pred_model_repeats,
        )

    @staticmethod
    def _get_cv_splitter(n_repeats, k_fold):
        if n_repeats < 1:
            cv_splitter = StratifiedKFold(n_splits=k_fold)
        else:
            cv_splitter = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats)

        return cv_splitter

    def _generate_folds(self, X: np.ndarray, y: np.ndarray) -> list:
        folds_list = []

        for fold_num, (train_indices, val_indices) in enumerate(self.cv_splitter.split(X, y)):
            suffix = f'Repeat_{fold_num // self.k_fold + 1}-Fold_{fold_num + 1}'

            fold_ctx = dict(
                model_name_suffix=suffix,
                train_indices=train_indices,
                val_indices=val_indices,
            )

            folds_list.append(fold_ctx)

        return folds_list

    def _construct_empty_oof(self, X: np.ndarray, y: np.ndarray):
        oof_pred_proba = np.zeros(shape=(len(X), len(np.unique(y))), dtype=np.float32)
        oof_pred_model_repeats = np.zeros(shape=len(X), dtype=np.uint8)

        return oof_pred_proba, oof_pred_model_repeats

    def _add_model_to_bag(self, model):
        self.models.append(model)

    def _update_oof(self, oof_pred_proba, oof_pred_model_repeats):
        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
            self._oof_pred_model_repeats = oof_pred_model_repeats

        else:
            self._oof_pred_proba += oof_pred_proba
            self._oof_pred_model_repeats += oof_pred_model_repeats

    def _concatenate(self, prev_data, new_data):
        return np.concatenate(prev_data, new_data.T, axis=-1)


class KFoldBaggingClassifier(BaseKFoldBagging):
    def __init__(self, model_base, n_repeats: int, k_fold: int, fold_fitting_strategy: str, n_jobs: int):
        super().__init__(
            model_base=model_base,
            n_repeats=n_repeats,
            k_fold=k_fold,
            fold_fitting_strategy=fold_fitting_strategy
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.unique_class = np.unique(y)
        folds_list = self._generate_folds(X=X, y=y)
        oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X, y=y)

        fold_fitting_strategy_args = self._get_fold_fitting_args(X, y, oof_pred_proba, oof_pred_model_repeats)
        fold_fitting_strategy = self.kfold_fitting_strategy(**fold_fitting_strategy_args)

        for fold_fit in folds_list:
            fold_fitting_strategy.schedule_fold_model_fit(fold_fit)

        fold_fitting_strategy.after_all_folds_scheduled()

        self._update_oof(oof_pred_proba, oof_pred_model_repeats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_proba = self.predict_proba(X)
        pred = np.argmax(pred_proba, axis=-1)

        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: Do analogue predict with model[0] and etc
        pred_proba = np.zeros(shape=(len(X), len(self.unique_class), 2), dtype=np.float32)

        for model in self.models:
            # model = self.load_child(model)
            pred_proba += model.predict_proba(X=X)

        pred_proba = pred_proba / len(self.models)

        return pred_proba


class KFoldBaggingRegressor(BaseKFoldBagging):
    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def predict_proba(self):
        raise NotImplementedError
