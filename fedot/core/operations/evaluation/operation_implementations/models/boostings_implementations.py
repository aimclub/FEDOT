import inspect
import math
import os
import time
from numbers import Real
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import early_stopping as lgbm_early_stopping
from lightgbm.callback import EarlyStopException as LGBMEarlyStopException
from matplotlib import pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utils import default_fedot_data_dir, is_multi_output_target
from fedot.core.repository.tasks import TaskTypesEnum


_XGBOOST_CALLBACKS_IN_FIT = 'callbacks' in inspect.signature(XGBClassifier.fit).parameters


class _FitTimeLimitCallback(TrainingCallback):
    """Stop XGBoost at a fit deadline and optionally compensate a short horizon.

    The learning-rate adjustment is deliberately opt-in.  It measures steady-state
    iteration throughput after a small warm-up and raises eta once only when the
    fit is projected to complete materially fewer trees than requested.
    """

    def __init__(self, seconds: Real, *, learning_rate=None, maximum_rounds=None,
                 adaptive_learning_rate=False):
        super().__init__()
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise ValueError('fit_time_limit must be a positive finite number')
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('fit_time_limit must be a positive finite number')
        self.seconds = seconds
        self.learning_rate = learning_rate
        self.maximum_rounds = maximum_rounds
        self.adaptive_learning_rate = bool(adaptive_learning_rate)
        self.deadline = None
        self.started_at = None
        self.iteration_starts = []
        self.projected_rounds = None
        self.adjusted_learning_rate = None

        if self.adaptive_learning_rate:
            if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
                raise ValueError('learning_rate must be a positive finite number')
            if not np.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError('learning_rate must be a positive finite number')
            if isinstance(maximum_rounds, bool) or not isinstance(maximum_rounds, Real):
                raise ValueError('maximum_rounds must be a positive integer')
            if not np.isfinite(maximum_rounds) or int(maximum_rounds) != maximum_rounds or maximum_rounds <= 0:
                raise ValueError('maximum_rounds must be a positive integer')
            self.learning_rate = float(learning_rate)
            self.maximum_rounds = int(maximum_rounds)

    def before_training(self, model):
        self.started_at = time.monotonic()
        self.deadline = self.started_at + self.seconds
        self.iteration_starts = []
        self.projected_rounds = None
        self.adjusted_learning_rate = None
        return model

    def before_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        del evals_log
        if not self.adaptive_learning_rate:
            return bool(
                epoch > 0
                and self.deadline is not None
                and time.monotonic() >= self.deadline
            )
        now = time.monotonic()
        if (
            epoch > 0
            and self.deadline is not None
            and now >= self.deadline
        ):
            return True
        self._adjust_learning_rate(model, epoch, now)
        return False

    def _adjust_learning_rate(self, model, epoch: int, now: float):
        if not self.adaptive_learning_rate:
            return
        self.iteration_starts.append(now)
        warmup_rounds = 16
        if epoch != warmup_rounds:
            return
        measured_round_times = np.diff(self.iteration_starts[4:])
        if measured_round_times.size == 0:
            return
        seconds_per_round = float(np.median(measured_round_times))
        if not np.isfinite(seconds_per_round) or seconds_per_round <= 0:
            return
        remaining_seconds = max(self.seconds - (now - self.started_at), 0.0)
        additional_rounds = int(remaining_seconds / (1.15 * seconds_per_round))
        self.projected_rounds = min(self.maximum_rounds, epoch + additional_rounds)
        if not warmup_rounds < self.projected_rounds <= 220:
            return
        target_learning_rate = self.learning_rate * math.sqrt(
            (self.maximum_rounds - warmup_rounds)
            / (self.projected_rounds - warmup_rounds)
        )
        target_learning_rate = min(0.2, target_learning_rate)
        if target_learning_rate <= self.learning_rate:
            return
        self.adjusted_learning_rate = target_learning_rate
        model.set_param({'eta': target_learning_rate})

    def after_training(self, model):
        self.deadline = None
        self.started_at = None
        return model


class _LightGBMFitTimeLimitCallback:
    """Interrupt LightGBM after a positive wall-clock fit allowance."""

    order = 10
    before_iteration = False

    def __init__(self, seconds: Real):
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise ValueError('fit_time_limit must be a positive finite number')
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('fit_time_limit must be a positive finite number')
        self.seconds = seconds
        self.deadline = time.monotonic() + seconds
        self.stopped = False

    def __call__(self, env):
        if env.iteration > 0 and time.monotonic() >= self.deadline:
            self.stopped = True
            raise LGBMEarlyStopException(
                env.iteration, env.evaluation_result_list
            )


class _CatBoostFitTimeLimitCallback:
    """Stop CatBoost after a positive wall-clock fit allowance."""

    def __init__(self, seconds: Real):
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise ValueError('fit_time_limit must be a positive finite number')
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('fit_time_limit must be a positive finite number')
        self.seconds = seconds
        self.deadline = None
        self.stopped = False

    def after_iteration(self, info) -> bool:
        now = time.monotonic()
        # CatBoost callbacks have no before-training hook. Iteration numbering
        # restarts at one for a repeated fit, which also resets this callback.
        if self.deadline is None or info.iteration <= 1:
            self.deadline = now + self.seconds
            self.stopped = False
            return True
        self.stopped = now >= self.deadline
        return not self.stopped


class FedotXGBoostImplementation(ModelImplementation):
    __operation_params = [
        'use_eval_set',
        'fit_time_limit',
        'fit_time_limit_adaptive_learning_rate',
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.check_and_update_params()

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        fit_time_limit = self.params.get('fit_time_limit')
        self.fit_callbacks = list(self.model_params.get('callbacks') or [])
        if fit_time_limit is not None:
            self.fit_callbacks.append(
                _FitTimeLimitCallback(
                    fit_time_limit,
                    learning_rate=self.model_params.get('learning_rate'),
                    maximum_rounds=self.model_params.get('n_estimators'),
                    adaptive_learning_rate=self.params.get(
                        'fit_time_limit_adaptive_learning_rate', False
                    ),
                )
            )
        if self.fit_callbacks:
            if _XGBOOST_CALLBACKS_IN_FIT:
                self.model_params.pop('callbacks', None)
            else:
                self.model_params['callbacks'] = self.fit_callbacks
        self.model = None
        self.features_names = None
        self.classes_ = None

    def fit(self, input_data: InputData):
        self.features_names = input_data.features_names

        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        if check_eval_set_condition(input_data, self.params):
            train_input, eval_input = train_test_data_setup(input_data)

            X_train, y_train = convert_to_dataframe(
                train_input, identify_cats=self.params.get('enable_categorical')
            )

            X_eval, y_eval = convert_to_dataframe(
                eval_input, identify_cats=self.params.get('enable_categorical')
            )

            self.model.eval_metric = self.set_eval_metric(self.classes_)

            self.model.fit(
                X=X_train, y=y_train,
                eval_set=[(X_eval, y_eval)],
                verbose=self.model_params['verbosity'],
                **self._fit_callback_params(),
            )
        else:
            # Disable parameter used for eval_set
            if bool(self.params.get('early_stopping_rounds')):
                self.model.early_stopping_rounds = None
                self.params.update(early_stopping_rounds=None)

            # Training model without splitting on train and eval
            X_train, y_train = convert_to_dataframe(
                input_data, identify_cats=self.params.get('enable_categorical')
            )
            self.model.fit(
                X=X_train, y=y_train,
                verbose=self.model_params['verbosity'],
                **self._fit_callback_params(),
            )

        return self.model

    def _fit_callback_params(self) -> dict:
        if _XGBOOST_CALLBACKS_IN_FIT and self.fit_callbacks:
            return {'callbacks': self.fit_callbacks}
        return {}

    def predict(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = convert_to_dataframe(input_data, self.params.get('enable_categorical'))
        prediction = self.model.predict(X)

        return prediction

    def check_and_update_params(self):
        early_stopping_rounds = self.params.get('early_stopping_rounds')
        use_eval_set = self.params.get('use_eval_set')

        if isinstance(early_stopping_rounds, int) and not use_eval_set:
            self.params.update(early_stopping_rounds=False)

        booster = self.params.get('booster')
        enable_categorical = self.params.get('enable_categorical')

        if booster == 'gblinear' and enable_categorical:
            self.params.update(enable_categorical=False)

        if booster == 'gbtree' and enable_categorical:
            self.params.update(enable_categorical=False)

    def get_feature_importance(self) -> list:
        return self.model.features_importances_

    def plot_feature_importance(self, importance_type='weight'):
        model_output = self.model.get_booster().get_score()
        features_names = self.features_names
        plot_feature_importance(features_names, model_output.values())

    @staticmethod
    def set_eval_metric(n_classes):
        if n_classes is None:  # if n_classes is None -> regression
            eval_metric = 'rmse'
        elif len(n_classes) < 3:  # if n_classes < 3 -> bin class
            eval_metric = 'auc'
        else:  # else multiclass
            eval_metric = 'mlogloss'

        return eval_metric


class FedotXGBoostClassificationImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = XGBClassifier(**self.model_params)

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = convert_to_dataframe(input_data, self.params.get('enable_categorical'))
        prediction = self.model.predict_proba(X)
        return prediction


class FedotXGBoostRegressionImplementation(FedotXGBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = XGBRegressor(**self.model_params)


class FedotLightGBMImplementation(ModelImplementation):
    __operation_params = ['use_eval_set', 'enable_categorical', 'fit_time_limit']

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.check_and_update_params()

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.model = None
        self.features_names = None
        self.classes_ = None

    def fit(self, input_data: InputData):
        self.features_names = input_data.features_names

        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        if check_eval_set_condition(input_data, self.params):
            train_input, eval_input = train_test_data_setup(input_data)

            X_train, y_train = convert_to_dataframe(
                train_input, identify_cats=self.params.get('enable_categorical')
            )

            X_eval, y_eval = convert_to_dataframe(
                eval_input, identify_cats=self.params.get('enable_categorical')
            )

            eval_metric = self.set_eval_metric(self.classes_)
            callbacks = self.update_callbacks()

            self.model.fit(
                X=X_train, y=y_train,
                eval_set=[(X_eval, y_eval)], eval_metric=eval_metric,
                callbacks=callbacks
            )
        else:
            # Disable parameter used for eval_set
            if bool(self.params.get('early_stopping_rounds')):
                self.model._other_params.update(early_stopping_rounds=None)
                self.params.update(early_stopping_rounds=None)

            if is_multi_output_target(input_data):
                self._convert_to_multi_output_model(input_data)

            # Training model without splitting on train and eval
            X_train, y_train = convert_to_dataframe(
                input_data, identify_cats=self.params.get('enable_categorical')
            )
            self.model.fit(
                X_train, y_train, callbacks=self.update_callbacks()
            )

        return self.model

    def predict(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = convert_to_dataframe(input_data, identify_cats=self.params.get('enable_categorical'))
        prediction = self.model.predict(X)

        return prediction

    def check_and_update_params(self):
        early_stopping_rounds = self.params.get('early_stopping_rounds')
        use_eval_set = self.params.get('use_eval_set')

        if isinstance(early_stopping_rounds, int) and not use_eval_set:
            self.params.update(early_stopping_rounds=None)

    def update_callbacks(self) -> list:
        callbacks = []

        esr = self.params.get('early_stopping_rounds')
        if isinstance(esr, int):
            callbacks.append(lgbm_early_stopping(esr, verbose=self.params.get('verbose')))

        fit_time_limit = self.params.get('fit_time_limit')
        if fit_time_limit is not None:
            callbacks.append(_LightGBMFitTimeLimitCallback(fit_time_limit))

        return callbacks

    @staticmethod
    def set_eval_metric(n_classes):
        if n_classes is None:  # if n_classes is None -> regression
            eval_metric = ''

        elif len(n_classes) < 3:  # if n_classes < 3 -> bin class
            eval_metric = 'binary_logloss'

        else:  # else multiclass
            eval_metric = 'multi_logloss'

        return eval_metric

    def plot_feature_importance(self):
        plot_feature_importance(self.features_names, self.model.feature_importances_)

    def _convert_to_multi_output_model(self, input_data: InputData):
        if input_data.task.task_type == TaskTypesEnum.classification:
            multiout_func = MultiOutputClassifier
        elif input_data.task.task_type in [TaskTypesEnum.regression, TaskTypesEnum.ts_forecasting]:
            multiout_func = MultiOutputRegressor
        else:
            raise ValueError(f"For task type '{input_data.task.task_type}' MultiOutput wrapper is not supported")

        self.model = multiout_func(self.model)

        return self.model


class FedotLightGBMClassificationImplementation(FedotLightGBMImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LGBMClassifier(**self.model_params)

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        X, _ = convert_to_dataframe(input_data, self.params.get('enable_categorical'))
        prediction = self.model.predict_proba(X)
        return prediction


class FedotLightGBMRegressionImplementation(FedotLightGBMImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LGBMRegressor(**self.model_params)


class FedotCatBoostImplementation(ModelImplementation):
    __operation_params = [
        'n_jobs', 'use_eval_set', 'enable_categorical',
        'callbacks', 'fit_time_limit',
    ]

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)

        self.check_and_update_params()

        self.model_params = {k: v for k, v in self.params.to_dict().items() if k not in self.__operation_params}
        self.fit_callbacks = list(self.params.get('callbacks') or [])
        fit_time_limit = self.params.get('fit_time_limit')
        if fit_time_limit is not None:
            self.fit_callbacks.append(_CatBoostFitTimeLimitCallback(fit_time_limit))
        self.model = None
        self.features_names = None

    def fit(self, input_data: InputData):
        self.features_names = input_data.features_names

        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        if check_eval_set_condition(input_data, self.params):
            # TODO: Using this method for tuning
            train_input, eval_input = train_test_data_setup(input_data)

            train_input = self.convert_to_pool(train_input, identify_cats=self.params.get('enable_categorical'))
            eval_input = self.convert_to_pool(eval_input, identify_cats=self.params.get('enable_categorical'))

            self.model.fit(
                X=train_input, eval_set=eval_input,
                **self._fit_callback_params(),
            )
        else:
            # Disable parameter used for eval_set
            if bool(self.params.get('use_best_model')):
                self.model._init_params.update(use_best_model=False)
                self.params.update(use_best_model=False)

            # Training model without splitting on train and eval
            train_input = self.convert_to_pool(
                input_data, identify_cats=self.params.get('enable_categorical')
            )
            self.model.fit(X=train_input, **self._fit_callback_params())

        return self.model

    def _fit_callback_params(self) -> dict:
        return {'callbacks': self.fit_callbacks} if self.fit_callbacks else {}

    def predict(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        prediction = self.model.predict(input_data.features)

        return prediction

    def check_and_update_params(self):
        n_jobs = self.params.get('n_jobs')
        self.params.update(thread_count=n_jobs)

        use_best_model = self.params.get('use_best_model')
        early_stopping_rounds = self.params.get('early_stopping_rounds')
        use_eval_set = self.params.get('use_eval_set')

        if (use_best_model or isinstance(early_stopping_rounds, int)) and not use_eval_set:
            self.params.update(use_best_model=False, early_stopping_rounds=False)

    @staticmethod
    def convert_to_pool(data: Optional[InputData], identify_cats: bool):
        return Pool(
            data=data.features,
            label=data.target,
            cat_features=data.categorical_idx if identify_cats else None,
            feature_names=data.features_names.tolist() if data.features_names is not None else None
        )

    def save_model(self, model_name: str = 'catboost'):
        save_path = os.path.join(default_fedot_data_dir(), f'catboost/{model_name}.cbm')
        self.model.save_model(save_path, format='cbm')

    def load_model(self, path):
        self.model = CatBoostClassifier()
        self.model.load_model(path)

    def get_feature_importance(self) -> (list, list):
        """ Return feature importance -> (feature_id (string), feature_importance (float)) """
        return self.model.get_feature_importance(prettified=True)

    def plot_feature_importance(self):
        plot_feature_importance(self.model.feature_names_, self.model.feature_importances_)


class FedotCatBoostClassificationImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostClassifier(**self.model_params)
        self.classes_ = None

    def fit(self, input_data: InputData):
        self.classes_ = np.unique(np.array(input_data.target))
        return super().fit(input_data=input_data)

    def predict_proba(self, input_data: InputData):
        if self.params.get('enable_categorical'):
            input_data = input_data.get_not_encoded_data()

        prediction = self.model.predict_proba(input_data.features)
        return prediction


class FedotCatBoostRegressionImplementation(FedotCatBoostImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = CatBoostRegressor(**self.model_params)


def plot_feature_importance(feature_names, feature_importance):
    fi = pd.DataFrame(index=feature_names)
    fi['importance'] = feature_importance

    fi.loc[fi['importance'] > 0.1].sort_values('importance').plot(
        kind='barh', figsize=(16, 9), title='Feature Importance')

    plt.show()


def convert_to_dataframe(data: Optional[InputData], identify_cats: bool):
    """
    Converts InputData data class to DataFrame.
    """
    features = pd.DataFrame(data=data.features)

    if data.target is not None and data.target.size > 0:
        if not is_multi_output_target(data):
            target = np.ravel(data.target[:features.shape[0]])
        else:
            target = data.target
    else:
        # TODO: temp workaround in case data.target is set to None intentionally
        #  for test.integration.models.test_model.check_predict_correct
        target = np.zeros(len(data.features))

    if identify_cats and data.categorical_idx is not None:
        for col in features.columns[data.categorical_idx]:
            features[col] = features[col].astype('category')

    if data.numerical_idx is not None:
        for col in features.columns[data.numerical_idx]:
            features[col] = features[col].astype('float')

    target = pd.DataFrame(target)

    return features, target


def check_eval_set_condition(input_data: InputData, params: OperationParameters) -> bool:
    """
    Checks the model training condition with eval_set.
    """
    is_using_eval_set = bool(params.get('use_eval_set'))
    if not is_using_eval_set or is_multi_output_target(input_data):
        return False

    # No special conditions for regression task
    if input_data.task.task_type == TaskTypesEnum.regression:
        return True

    # For classification task check
    # if all classes presented in train_set are also presented in eval_set
    if input_data.task.task_type == TaskTypesEnum.classification:
        train_input, eval_input = train_test_data_setup(input_data)
        train_classes = np.unique(train_input.target)
        eval_classes = np.unique(eval_input.target)
        all_classes_present_in_eval = np.all(np.isin(train_classes, eval_classes))
        if all_classes_present_in_eval:
            return True

    return False
