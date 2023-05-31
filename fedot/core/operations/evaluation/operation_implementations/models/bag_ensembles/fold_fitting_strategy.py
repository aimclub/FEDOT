import copy
from abc import abstractmethod, ABC


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
    def _fit(self, fold_ctx):
        """
        Method is called when a fold is ready to be fit
        """


class FoldFittingStrategy(AbstractFoldFittingStrategy, ABC):
    """ Provides some default implementation for AbstractFoldFittingStrategy

    Args:
        model_base: The template for the folds of model to be trained on.
        X: 'np.ndarray' The training data the model will be trained on.
        y: 'np.ndarray' The target value of the training data.
        bagged_ensemble_model: 'BaseKFoldBagging' The ensemble model that holds all the trained folds.
        oof_pred_proba: 'np.ndarray' Out of folds predict probabilities that are already calculated.
        oof_pred_model_repeats: 'np.ndarray' Number of repeats the out of folds predict probabilities has been done.

    TODO:
        time_start: 'float32'
        time_limit: 'float32'
        save_folds: 'bool' saving indices of folds into disk

    """

    def __init__(self, model_base, X, y, bagged_ensemble_model, oof_pred_proba, oof_pred_model_repeats):
        self.model_base = model_base

        self.X = X
        self.y = y

        self.bagged_ensemble_model = bagged_ensemble_model

        self.oof_pred_proba = oof_pred_proba
        self.oof_pred_model_repeats = oof_pred_model_repeats

        self.jobs = []

    def schedule_fold_model_fit(self, fold_ctx):
        raise NotImplementedError

    def after_all_folds_scheduled(self):
        raise NotImplementedError

    def _update_bagged_ensemble(self, fold_model, pred_proba, fold_ctx):
        val_indices = fold_ctx['val_indices']

        # TODO: Saving fold model
        # self.save(fold_model)

        self.oof_pred_proba[val_indices] += pred_proba
        self.oof_pred_model_repeats[val_indices] += 1
        self.bagged_ensemble_model.add_model_to_bag(model=fold_model)

    def _predict_oof(self, fold_model, fold_ctx):
        val_indices = fold_ctx.get('val_indices')

        X_val_fold = self.X[val_indices]
        # y_val_fold = self.y[val_indices]

        pred_proba = fold_model.predict_proba(X_val_fold)

        # TODO: Save val score metric into model or logger
        # labels = np.argmax(pred_proba, axis=-1)
        # fold_model.score(y_true=y_val_fold, y_score=pred)

        # TODO: Remove model to reduce RAM memory
        # self.reduce_memory(fold_model)

        return fold_model, pred_proba

    def _get_by_val_indices(self, indices):
        return self.X[indices], self.y[indices]


class SequentialFoldFittingStrategy(FoldFittingStrategy):
    """ Strategy for fitting models in a sequence """
    def __init__(self, **kwargs):
        super(SequentialFoldFittingStrategy, self).__init__(**kwargs)

    def schedule_fold_model_fit(self, fold_ctx):
        self.jobs.append(fold_ctx)

    def after_all_folds_scheduled(self):
        """ Create all """
        for job in self.jobs:
            self._fit_fold_model(job)

    def _fit_fold_model(self, fold_ctx):
        fold_model = self._fit(fold_ctx)
        fold_model, pred_proba = self._predict_oof(fold_model, fold_ctx)
        self._update_bagged_ensemble(fold_model, pred_proba, fold_ctx)

    def _fit(self, fold_ctx):
        train_indices = fold_ctx.get('train_indices')
        val_indices = fold_ctx.get('val_indices')

        X_fold, y_fold = self._get_by_val_indices(train_indices)
        X_val_fold, y_val_fold = self._get_by_val_indices(val_indices)

        fold_model = copy.deepcopy(self.model_base)

        fold_model.fit(X=X_fold, y=y_fold, eval_set=(X_val_fold, y_val_fold))

        return fold_model


class ParallelFoldFittingStrategy(FoldFittingStrategy):
    """ Strategy for fitting models in a parallel """
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

    def _fit(self, fold_ctx):
        raise NotImplementedError