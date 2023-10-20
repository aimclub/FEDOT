import copy
import time
from abc import abstractmethod
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from fedot.core.data.cv_folds import cv_generator
from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.evaluation.operation_implementations.models.bag_ensembles.fold_fitting_strategy import \
    SequentialFoldFittingStrategy, ParallelFoldFittingStrategy
from fedot.core.operations.operation_parameters import OperationParameters


class BaseKFoldBagging:
    """ Base k-fold n-repeated bagging ensemble class which provides some default implementation for specific
        objects for each task.

        Bagged ensemble fits a given base model multiple times (n-repeats) across different splits (k-fold)
        of the training data. Each model produce out-of-fold prediction and all probs or values stack into
        single prediction vector. This vector could be used in meta-model or fit another layer of bagged ensemble.

        Args:
            model_base: `ModelImplementation` the base model
            k_fold: 'int' the number of data splits and base estimators in bagging ensembles
            n_repeats: 'int' the number of fold fitting repeats per each estimator
            fold_fitting_strategy: 'str' the fitting strategy
            .. details:: possible strategy:
                - ``sequential`` 'SequentialFoldFittingStrategy' each models in bagging layer fitting in the order of the queue
                - ``parallel`` 'ParallelFoldFittingStrategy' the whole fitting process is divided between processors

            n_jobs: the number of jobs to run in parallel for parallel strategy.

    """
    @abstractmethod
    def __init__(self, estimator: ModelImplementation, k_fold: int = 1, n_repeats: int = 1, fold_fitting_strategy: str = 'sequential', n_jobs: int = 1, random_seed: int = 42):
        self.model_base = estimator
        # Store special
        self.models = []

        self.k_fold = k_fold
        self.n_repeats = n_repeats

        self._oof_pred_proba = None
        self._oof_pred_model_repeats = None

        self.X_new = None
        self.unique_class = None
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.fold_fitting_strategy = fold_fitting_strategy
        self.kfold_fitting_strategy = self._get_fold_fitting_strategy(self.fold_fitting_strategy)

    @staticmethod
    def _get_fold_fitting_strategy(fold_fitting_strategy):
        if fold_fitting_strategy == 'sequential':
            fold_fitting_strategy = SequentialFoldFittingStrategy

        # TODO: parallel for cpu and gpu
        elif fold_fitting_strategy == 'parallel':
            fold_fitting_strategy = ParallelFoldFittingStrategy

        else:
            raise ValueError(f'{fold_fitting_strategy} is unknown fitting strategy')

        return fold_fitting_strategy

    def _get_fold_fitting_args(self, oof_pred_proba: np.ndarray,
                               oof_pred_model_repeats: np.ndarray) -> dict:
        return dict(
            model_base=self.model_base,
            bagged_ensemble_model=self,
            oof_pred_proba=oof_pred_proba,
            oof_pred_model_repeats=oof_pred_model_repeats,
            n_jobs=self.n_jobs,
        )

    def _generate_folds(self, train_data: InputData) -> list:
        folds_list = []
        cv = cv_generator(
            data=train_data,
            cv_folds=self.k_fold,
            n_repeats=self.n_repeats,
            random_seed=self.random_seed,
            return_indices=True
        )

        for fold_num, (train, val, train_indices, val_indices) in enumerate(cv):
            suffix = f'Repeat_{fold_num // self.k_fold + 1}-Fold_{fold_num + 1}'

            fold_ctx = dict(
                model_name_suffix=suffix,
                train_data=train,
                val_data=val,
                train_indices=train_indices,
                val_indices=val_indices
            )

            folds_list.append(fold_ctx)

        return folds_list

    @staticmethod
    def _construct_empty_oof(train_data):
        X, y = train_data.features, train_data.target

        oof_pred_proba = np.zeros(shape=(len(X), len(np.unique(y))), dtype=np.float32)
        oof_pred_model_repeats = np.zeros(shape=len(X), dtype=np.uint8)

        return oof_pred_proba, oof_pred_model_repeats

    @staticmethod
    def _concatenate(prev_data, new_data):
        return np.concatenate(prev_data, new_data.T, axis=-1)

    def add_model_to_bag(self, model):
        self.models.append(model)

    def _update_oof(self, oof_pred_proba, oof_pred_model_repeats):
        if self._oof_pred_proba is None:
            self._oof_pred_proba = oof_pred_proba
            self._oof_pred_model_repeats = oof_pred_model_repeats

        else:
            self._oof_pred_proba += oof_pred_proba
            self._oof_pred_model_repeats += oof_pred_model_repeats


class KFoldBaggingClassifier(BaseKFoldBagging):
    def __init__(self, estimator: ModelImplementation, k_fold: int, n_repeats: int, fold_fitting_strategy: str, n_jobs: int):
        super().__init__(
            estimator=estimator,
            n_repeats=n_repeats,
            k_fold=k_fold,
            fold_fitting_strategy=fold_fitting_strategy,
            n_jobs=n_jobs
        )

    def fit(self, train_data: InputData):
        folds_list = self._generate_folds(train_data)
        oof_pred_proba, oof_pred_model_repeats = self._construct_empty_oof(train_data)

        fold_fitting_strategy_args = self._get_fold_fitting_args(oof_pred_proba, oof_pred_model_repeats)
        fold_fitting_strategy = self.kfold_fitting_strategy(**fold_fitting_strategy_args)

        for fold_fit in folds_list:
            fold_fitting_strategy.schedule_fold_model_fit(fold_fit)

        fold_fitting_strategy.after_all_folds_scheduled()

        self._update_oof(oof_pred_proba, oof_pred_model_repeats)

    def predict(self, input_data: InputData) -> np.ndarray:
        pred_proba = self.predict_proba(input_data)
        pred = np.argmax(pred_proba, axis=-1)

        return pred

    def predict_proba(self, input_data: InputData) -> np.ndarray:
        model = self.models[0]
        pred_proba = model.predict_proba(input_data)

        for model in self.models[1:]:
            # model = self.load_child(model)
            pred_proba += model.predict_proba(input_data)

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
