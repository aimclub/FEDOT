from typing import Callable, Optional, Tuple, Union

from numpy.random import randint, choice as nprand_choice
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from skopt import BayesSearchCV

from core.models.data import InputData
from core.models.data import train_test_data_setup


class Tuner:
    def tune(self, trained_model, tune_data: InputData, params_range: dict, iterations: int,
             cv_fold_num: int, scorer: Union[str, callable]):
        raise NotImplementedError()


class SklearnTuner(Tuner):
    def __init__(self):
        self.search_strategy = None

    def tune(self, trained_model, tune_data: InputData, params_range: dict, iterations: int,
             cv_fold_num: int, scorer: Union[str, callable]):
        raise NotImplementedError()

    def _sklearn_tune(self, search_strategy, tune_data: InputData):
        try:
            search = search_strategy.fit(tune_data.features, tune_data.target.ravel())
            return search.best_params_, search.best_estimator_
        except ValueError as ex:
            print(f'Unsuccessful fit because of {ex}')
            return None


class SklearnRandomTuner(SklearnTuner):
    def tune(self, trained_model, tune_data: InputData, params_range: dict, iterations: int,
             cv_fold_num: int, scorer: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        self.search_strategy = RandomizedSearchCV(estimator=trained_model,
                                                  param_distributions=params_range,
                                                  n_iter=iterations,
                                                  cv=cv_fold_num,
                                                  scoring=scorer)
        return self._sklearn_tune(search_strategy=self.search_strategy,
                                  tune_data=tune_data)


class SklearnGridSearchTuner(SklearnTuner):
    def tune(self, trained_model, tune_data: InputData, params_range: dict, iterations: int,
             cv_fold_num: int, scorer: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        self.search_strategy = GridSearchCV(estimator=trained_model,
                                            param_grid=params_range,
                                            cv=cv_fold_num,
                                            scoring=scorer)
        return self._sklearn_tune(GridSearchCV, tune_data)


class SklearnBayesSearchCV(SklearnTuner):
    def tune(self, trained_model, tune_data: InputData, params_range: dict, iterations: int,
             cv_fold_num: int, scorer: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        self.search_strategy = BayesSearchCV(estimator=trained_model,
                                             search_spaces=params_range,
                                             n_iter=iterations,
                                             cv=cv_fold_num,
                                             scoring=scorer)
        return self._sklearn_tune(BayesSearchCV, tune_data)


class SklearnCustomRandomTuner(Tuner):
    def tune(self, trained_model, tune_data: InputData, params_range: dict, iterations: int,
             cv_fold_num: int, scorer: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        try:
            best_score = scorer(estimator=trained_model, X=tune_data.features, y_true=tune_data.target)
            best_model = trained_model
            best_params = None
            for i in range(iterations):
                params = {k: nprand_choice(v) for k, v in params_range.items()}
                for param in params:
                    setattr(trained_model, param, params[param])
                score = cross_val_score(trained_model, tune_data.features, tune_data.target, scoring=scorer,
                                        cv=cv_fold_num).mean()
                if score > best_score:
                    best_params = params
                    best_model = trained_model
                    best_score = score

            return best_params, best_model
        except ValueError as ex:
            print(f'Unsuccessful fit because of {ex}')
            return None


class ForecastingCustomRandomTuner:
    # TODO discuss
    def tune(self,
             fit: Callable,
             predict: Callable,
             tune_data: InputData, params_range: dict,
             default_params: dict, iterations: int) -> dict:

        tune_train_data, tune_test_data = train_test_data_setup(tune_data, 0.5)

        trained_model_default = fit(tune_test_data, default_params)
        prediction_default = predict(trained_model_default, tune_test_data)
        best_quality_metric = _regression_prediction_quality(prediction=prediction_default,
                                                             real=tune_test_data.target)
        best_params = default_params

        for _ in range(iterations):
            random_params = get_random_params(params_range)
            try:
                trained_model_candidate = fit(tune_train_data, random_params)
                prediction_candidate = predict(trained_model_candidate,
                                               tune_test_data)
                quality_metric = _regression_prediction_quality(prediction=prediction_candidate,
                                                                real=tune_test_data.target)
                if quality_metric < best_quality_metric:
                    best_params = random_params
            except ValueError:
                pass
        return best_params


def get_random_params(params_range):
    candidate_params = {}
    for param in params_range:
        param_range = params_range[param]
        param_range_left, param_range_right = param_range[0], param_range[1]
        if isinstance(param_range_left, tuple):
            # set-based params with constant length
            candidate_param = get_constant_length_range(param_range_left, param_range_right)
        elif isinstance(param_range_left, list):
            # set-based params with varied length
            candidate_param = get_varied_length_range(param_range_left, param_range_right)
        else:
            raise ValueError(f'Un-supported params range type {type(param_range_left)}')
        candidate_params[param] = candidate_param
    return candidate_params


def get_constant_length_range(left_range, right_range):
    candidate_param = []
    for sub_param_ind in range(len(left_range)):
        new_sub_param = randint(left_range[sub_param_ind],
                                right_range[sub_param_ind] + 1)
        candidate_param.append(new_sub_param)
    return tuple(candidate_param)


def get_varied_length_range(left_range, right_range):
    candidate_param = []
    subparams_num = randint(1, len(right_range))
    for sub_param_ind in range(subparams_num):
        new_sub_param = randint(left_range[sub_param_ind],
                                right_range[sub_param_ind] + 1)
        candidate_param.append(new_sub_param)
    return candidate_param


def _regression_prediction_quality(prediction, real):
    return mse(y_true=real, y_pred=prediction, squared=False)
