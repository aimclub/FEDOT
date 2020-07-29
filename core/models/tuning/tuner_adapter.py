from hyperopt import tpe, fmin, Trials, space_eval
from sklearn.model_selection import cross_val_score
import random
import numpy as np


def _search_space_transform(params):
    space = {}
    for param_name, param_space in params.items():
        space[param_name] = hp.choice(param_name, param_space)
    return space


class TunerAdapter:
    def __init__(self, tuner: 'Tuner'):
        raise NotImplementedError()


class HyperoptAdapter(TunerAdapter):
    def __init__(self, tuner: 'TPE'):
        super(HyperoptAdapter, self).__init__(tuner)
        self.model_to_adapt = tuner.trained_model
        self.data = tuner.tune_data
        self.params = _search_space_transform(tuner.params_range)
        self.scorer = tuner.scorer
        self.best_metric = 0
        self.best_params: dict = dict()

    def _objective_function(self, params):
        self.model_to_adapt.set_params(**params)
        metric = cross_val_score(self.model_to_adapt,
                                 self.data.features,
                                 self.data.target,
                                 cv=5, scoring=self.scorer).mean()

        if self.data.task.task_type.name == 'classification':
            return 1 - metric.mean()
        else:
            return metric.mean()

    def tune(self, iterations=100):
        hyperopt_trials = Trials()
        best_encrypted = fmin(fn=self._objective_function,
                              space=self.params, algo=tpe.suggest,
                              max_evals=iterations,
                              trials=hyperopt_trials)
        self.best_params = space_eval(space=self.params,
                                      hp_assignment=best_encrypted)
        return self.best_params, self.best_model

    @property
    def best_model(self):
        return self.model_to_adapt.set_params(**self.best_params)
