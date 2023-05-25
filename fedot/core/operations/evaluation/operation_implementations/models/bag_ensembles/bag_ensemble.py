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
                 base_model,
                 n_layers: int = 1,
                 n_repeats: int = 0,
                 k_fold: int = 5,
                 time_limit: float = None,
                 fold_fitting_strategy: str = None
                 ):
        self.base_model = base_model
        self.layers = [[] for _ in range(n_layers)]

        self.n_layers = n_layers
        self.n_repeated = n_repeats
        self.k_fold = k_fold

        self.time_limit = time_limit
        self.time_start = None

        self._oof_pred_proba = None
        self._oof_pred_model_repeats = None

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

    def _get_fold_fitting_args(self, X, y, oof_pred_proba, oof_pred_model_repeats):
        return dict(
            model_base=self.model_base,
            bagged_ensemble_model=self,
            X=X,
            y=y,
            time_limit=self.time_limit,
            time_start=self.time_start,
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

    def _generate_folds(self, X: np.ndarray, y: np.ndarray, k_fold_start, k_fold_end, n_repeat_start, n_repeat_end) -> (
            list, int, int):
        k_fold = self.cv_splitter.n_splits
        kfolds = self.cv_splitter.split(X=X, y=y)

        fold_start = n_repeat_start * k_fold + k_fold_start
        fold_end = (n_repeat_end - 1) * k_fold + k_fold_end
        folds_to_fit = fold_end - fold_start

        fold_fit_args_list = []
        n_repeats_started = 0
        n_repeats_finished = 0

        for repeat in range(n_repeat_start, n_repeat_end):  # For each repeat
            is_first_set = repeat == n_repeat_start
            is_last_set = repeat == (n_repeat_end - 1)

            if (not is_first_set) or (k_fold_start == 0):
                n_repeats_started += 1

            fold_in_set_start = k_fold_start if repeat == n_repeat_start else 0
            fold_in_set_end = k_fold_end if is_last_set else k_fold

            for fold_in_set in range(fold_in_set_start, fold_in_set_end):  # For each fold
                fold = fold_in_set + (repeat * k_fold)
                fold_ctx = dict(
                    model_name_suffix=f'S{repeat + 1}F{fold_in_set + 1}',  # S5F3 = 3rd fold of the 5th repeat set
                    fold=kfolds[fold],
                    is_last_fold=fold == (fold_end - 1),
                    folds_to_fit=folds_to_fit,
                    folds_finished=fold - fold_start,
                    folds_left=fold_end - fold,
                )

                fold_fit_args_list.append(fold_ctx)
            if fold_in_set_end == k_fold:
                n_repeats_finished += 1

        return fold_fit_args_list, n_repeats_started, n_repeats_finished

    def _construct_empty_oof(self, X: np.ndarray, y: np.ndarray):
        oof_pred_proba = np.zeros(shape=(len(X), len(np.unique(y))), dtype=np.float32)
        oof_pred_model_repeats = np.zeros(shape=len(X), dtype=np.uint8)

        return oof_pred_proba, oof_pred_model_repeats

    def _update_oof(self, oof_pred_proba, oof_pred_model_repeats):
        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
            self._oof_pred_model_repeats = oof_pred_model_repeats

        else:
            self._oof_pred_proba += oof_pred_proba
            self._oof_pred_model_repeats += oof_pred_model_repeats


class KFoldBaggingClassifier(BaseKFoldBagging):
    def __init__(self, base_model: object, n_layers: int, n_repeats: int, k_fold: int):
        super().__init__(base_model=base_model, n_layers=n_layers, n_repeats=n_repeats, k_fold=k_fold)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.time_start = time.time()

        for layer in self.layers:
            models, predictions = self._fit_layer(X, y)
            layer.append(models)
            X = self.concatenate(X, predictions)

        return self

    def _fit_layer(self, X, y):
        fold_fit_args_list, n_repeats_started, n_repeats_finished = self._generate_folds(X=X, y=y)

        oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(X=X, y=y)

        fold_fitting_strategy_args = self._get_fold_fitting_args(X, y, oof_pred_proba, oof_pred_model_repeats)
        fold_fitting_strategy = self.kfold_fitting_strategy(**fold_fitting_strategy_args)

        for fold_fit_args in fold_fit_args_list:
            fold_fitting_strategy.schedule_fold_model_fit(**fold_fit_args)

        fold_fitting_strategy.after_all_folds_scheduled()

        for model in models:
            self.add_child(model=model)

        self._update_oof(oof_pred_proba, oof_pred_model_repeats)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, predict_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KFoldBaggingRegressor(BaseKFoldBagging):
    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def predict_proba(self):
        raise NotImplementedError
