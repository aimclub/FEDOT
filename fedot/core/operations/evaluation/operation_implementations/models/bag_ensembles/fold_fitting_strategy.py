import copy
import time
from abc import abstractmethod


class AbstractFoldFittingStrategy:
    @abstractmethod
    def schedule_fold_model_fit(self, fold_ctx):
        """
        Schedule fold model training.
        By design this part is supposed to be 'lazy' evaluator,
        no actual training is performed here.
        Distributed fitters will handle jobs scheduling here.
        """

    @abstractmethod
    def after_all_folds_scheduled(self):
        """
        Method is called when all the folds are scheduled.
        Local fitters will perform training here.
        Distributed fitters will handle job handles and results retrieval here.
        """

    @abstractmethod
    def _fit(self, time_start_fold, time_limit_fold, fold_ctx, kwargs):
        """
        Method is called when a fold is ready to be fit
        """


class FoldFittingStrategy(AbstractFoldFittingStrategy):
    def __init__(self,
                 model_base,
                 model_base_kwargs,
                 X,
                 y,
                 time_limit,
                 time_start,
                 bagged_ensemble_model,
                 oof_pred_proba,
                 oof_pred_model_repeats
    ):
        self.model_base = model_base
        self.model_base_kwargs = model_base_kwargs
        self.X = X
        self.y = y
        self.time_limit = time_limit
        self.time_start = time_start
        self.bagged_ensemble_model = bagged_ensemble_model
        self.oof_pred_proba = oof_pred_proba
        self.oof_pred_model_repeats = oof_pred_model_repeats
        self.jobs = []

    def schedule_fold_model_fit(self, fold_ctx):
        raise NotImplementedError

    def after_all_folds_scheduled(self):
        raise NotImplementedError

    def _update_bagged_ensemble(self, fold_model, pred_proba, fold_ctx):
        _, val_index = fold_ctx['fold']

        # TODO: Saving fold model

        self.oof_pred_proba[val_index] += pred_proba
        self.oof_pred_model_repeats[val_index] += 1
        self.bagged_ensemble_model._add_model_to_bag(model=fold_model)

    def _predict_oof(self, fold_model, fold_ctx):
        fold, _ = self._get_fold_properties(fold_ctx)
        _, val_index = fold

        X_val_fold = self.X[val_index]
        y_val_fold = self.y[val_index]

        pred_proba = fold_model.predict_proba(X_val_fold)
        fold_model.val_score = fold_model.score(y_true=y_val_fold, y_pred_proba=pred_proba)

        # TODO: Remove model to reduce RAM memory

    def _get_fold_time_limit(self, fold):


    @staticmethod
    def _get_fold_properties(fold_ctx):
        return fold_ctx


class SequentialFoldFittingStrategy(FoldFittingStrategy):
    def __init__(self, **kwargs):
        super(SequentialFoldFittingStrategy, self).__init__(**kwargs)

    def schedule_fold_model_fit(self, fold_ctx):
        self.jobs.append(fold_ctx)

    def after_all_folds_scheduled(self):
        for job in self.jobs:
            self._fit_fold_model(job)

    def _fit_fold_model(self, fold_ctx):
        time_start_fold = time.time()
        time_limit_fold = self._get_fold_time_limit(fold_ctx)

        fold_model = self._fit(time_start_fold, time_limit_fold, fold_ctx, self.model_base_kwargs)
        fold_model, pred_proba = self._predict_oof(fold_model, fold_ctx)
        self._update_bagged_ensemble(fold_model, pred_proba, fold_ctx)

    def _fit(self, time_start_fold, time_limit_fold, fold_ctx, kwargs):
        fold, _, _, _, _, model_name_suffix = self._get_fold_properties(fold_ctx)
        train_index, val_index = fold

        X_fold, X_val_fold = self.X[train_index], self.X[val_index]
        y_fold, y_val_fold = self.y[train_index], self.y[val_index]

        fold_model = copy.deepcopy(model_base)
        fold_model.name = f'{fold_model.name}{model_name_suffix}'

        fold_model.fit(
            X=X_fold, y=y_fold,
            X_val=X_val_fold, y_val_fold=y_val_fold,
            time_limit=time_limit_fold
        )

        fold_model.fit_time = time.time() - time_start_fold
        return fold_model


class ParallelFoldFittingStrategy(FoldFittingStrategy):
    def __init__(self, **kwargs):
        super(ParallelFoldFittingStrategy, self).__init__(**kwargs)
        self.max_memory_usage = None
        self.time_start_fit = None
        self.time_end_fit = None
        self.fit_time = None
        self.predict_time = None
        self.resources, self.batches, self.num_parralle_jobs = None, None, None

    def schedule_fold_model_fit(self, fold_ctx):
        self.jobs.append(fold_ctx)

    def after_all_folds_scheduled(self):
        raise NotImplementedError

    def _fit(self):
        raise NotImplementedError