from copy import deepcopy

from hyperopt import tpe, fmin, Trials, space_eval, hp
from sklearn.model_selection import cross_val_score


def _search_space_transform(params):
    space = {}
    for param_name, param_space in params.items():
        space[param_name] = hp.choice(param_name, param_space)
    return space


class TunerAdapter:
    def __init__(self, tuner: 'Tuner'):
        self.tuner = tuner
        self.model_to_adapt = deepcopy(tuner.trained_model)
        self.data = tuner.tune_data
        self.params = _search_space_transform(tuner.params_range)
        self.scorer = tuner.scorer
        self.best_params: dict = dict()


class HyperoptAdapter(TunerAdapter):
    def __init__(self, tuner: 'TPE'):
        super(HyperoptAdapter, self).__init__(tuner)

    def _objective_function(self, params):
        self.model_to_adapt.set_params(**params)
        metric = cross_val_score(self.model_to_adapt,
                                 self.data.features,
                                 self.data.target,
                                 cv=5, scoring=self.scorer).mean()

        if self.greater_is_better():
            return -metric
        else:
            return metric

    def tune(self, iterations: int = 100, timeout_sec: int = 60):
        hyperopt_trials = Trials()
        best_encrypted = fmin(fn=self._objective_function,
                              space=self.params, algo=tpe.suggest,
                              max_evals=iterations,
                              trials=hyperopt_trials,
                              timeout=timeout_sec)
        self.best_params = space_eval(space=self.params,
                                      hp_assignment=best_encrypted)
        return self.best_params, self.best_model

    @property
    def best_model(self):
        return self.model_to_adapt.set_params(**self.best_params)

    def greater_is_better(self):
        """
        Extracting the private field of scorer to check
        whether the metric has a property of greater_is_better
        """
        is_greater = True if self.scorer._sign == 1 else False
        return is_greater
