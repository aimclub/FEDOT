import random
from random import uniform
from typing import Callable, Optional, Tuple, Union, TYPE_CHECKING

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from skopt import BayesSearchCV

from core.models.data import InputData
from core.models.data import train_test_data_setup

if TYPE_CHECKING:
    from core.models.evaluation.evaluation import EvaluationStrategy


class Tuner:
    def tune(self, eval_strategy: 'EvaluationStrategy', tune_data: InputData, params_range: dict, iterations: int,
             cv: int, scoring: Union[str, callable]):
        raise NotImplementedError()


class SklearnTuner(Tuner):
    def tune(self, eval_strategy: 'EvaluationStrategy', tune_data: InputData, params_range: dict, iterations: int,
             cv: int, scoring: Union[str, callable]):
        raise NotImplementedError()

    def _sklearn_tune(self, search_strategy, eval_strat: 'EvaluationStrategy', tune_data: InputData,
                      params_range: dict, iterations: int, cv: int, scoring: Union[str, callable]):
        estimator = eval_strat.fit(train_data=tune_data)

        try:
            if search_strategy is GridSearchCV:
                clf = search_strategy(estimator, params_range, cv=cv, scoring=scoring)
            else:
                clf = search_strategy(estimator, params_range, n_iter=iterations, cv=cv, scoring=scoring)

            search = clf.fit(tune_data.features, tune_data.target.ravel())

            return search.best_params_, search.best_estimator_
        except ValueError as ex:
            print(f'Unsuccessful fit because of {ex}')
            return None


class SklearnRandomTuner(SklearnTuner):
    def tune(self, eval_strategy: 'EvaluationStrategy', tune_data: InputData, params_range: dict, iterations: int,
             cv: int, scoring: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        return self._sklearn_tune(RandomizedSearchCV, eval_strategy, tune_data,
                                  params_range, iterations, cv, scoring)


class SklearnGridSearchTuner(SklearnTuner):
    def tune(self, eval_strategy: 'EvaluationStrategy', tune_data: InputData, params_range: dict, iterations: int,
             cv: int, scoring: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        return self._sklearn_tune(GridSearchCV, eval_strategy, tune_data,
                                  params_range, iterations, cv, scoring)


class SklearnBayesSearchCV(SklearnTuner):
    def tune(self, eval_strategy: 'EvaluationStrategy', tune_data: InputData, params_range: dict, iterations: int,
             cv: int, scoring: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        return self._sklearn_tune(BayesSearchCV, eval_strategy, tune_data,
                                  params_range, iterations, cv, scoring)


class SklearnCustomRandomTuner(Tuner):
    def tune(self, eval_strategy: 'EvaluationStrategy', tune_data: InputData, params_range: dict, iterations: int,
             cv: int, scoring: Union[str, callable]) -> (Optional[Tuple[dict, object]]):
        try:
            trained_model = eval_strategy.fit(tune_data)
            best_score = scoring(estimator=trained_model, X=tune_data.features, y_true=tune_data.target)
            best_model = trained_model
            best_params = None
            for i in range(iterations):
                params = {k: random.choice(v) for k, v in params_range.items()}
                for param in params:
                    setattr(trained_model, param, params[param])
                score = cross_val_score(trained_model, tune_data.features, tune_data.target, scoring=scoring,
                                        cv=cv).mean()
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
            random_params = _get_random_params(params_range)
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


def _get_random_params(params_range):
    candidate_params = {}
    for param in params_range:
        param_range = params_range[param]
        param_range_left, param_range_right = param_range[0], param_range[1]
        if isinstance(param_range_left, tuple):
            # set-based params with constant length
            candidate_param = list(param_range_left)
            for sub_param_ind in range(len(candidate_param)):
                candidate_param[sub_param_ind] = int(round(uniform(param_range_left[sub_param_ind],
                                                                   param_range_right[sub_param_ind])))
            candidate_param = tuple(candidate_param)
        elif isinstance(param_range_left, list):
            # set-based params with varied length
            candidate_param = []
            subparams_num = uniform(1, len(param_range_right))
            for sub_param_ind in range(subparams_num):
                new_sub_param = int(round(uniform(param_range_left[sub_param_ind],
                                                  param_range_right[sub_param_ind])))
                candidate_param.append(new_sub_param)
        else:
            raise ValueError(f'Un-supported params range {type(param_range_left)}')
        candidate_params[param] = candidate_param
    return candidate_params


def _regression_prediction_quality(prediction, real):
    return mse(y_true=real, y_pred=prediction, squared=False)
