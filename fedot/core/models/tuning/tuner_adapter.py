from copy import deepcopy
import numpy as np

from hyperopt import tpe, fmin, Trials, space_eval, hp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as roc_auc
from fedot.core.repository.tasks import TaskTypesEnum


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

        # train test split
        input_features = np.array(self.data.features)
        input_target = np.array(self.data.target)
        x_train, x_test, y_train, y_test = train_test_split(input_features,
                                                            input_target,
                                                            train_size=0.6,
                                                            random_state=1)
        try:
            self.model_to_adapt.fit(x_train, y_train)
            predicted = self.model_to_adapt.predict(x_test)

            if self.data.task.task_type == TaskTypesEnum.regression:
                metric = mean_absolute_error(predicted, y_test)
            elif self.data.task.task_type == TaskTypesEnum.classification:
                metric = roc_auc(predicted, y_test, multi_class='ovr')
        except ValueError:
            if self.greater_is_better():
                metric = -999999.0
            else:
                metric = 999999.0

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
