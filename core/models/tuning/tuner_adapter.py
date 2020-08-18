import timeit
from abc import abstractmethod
from copy import deepcopy
from random import sample

import numpy as np
from hyperopt import tpe, fmin, Trials, space_eval, hp
from scipy import stats
from sklearn.model_selection import cross_val_score, GroupKFold


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
        self.scorer = tuner.scorer
        self.best_params: dict = dict()

    @abstractmethod
    def tune(self):
        raise NotImplementedError()

    @property
    def best_model(self):
        return self.model_to_adapt.set_params(**self.best_params)


class HyperoptAdapter(TunerAdapter):
    def __init__(self, tuner: 'TPE'):
        super(HyperoptAdapter, self).__init__(tuner)
        self.params = _search_space_transform(tuner.params_range)

    def objective_function(self, params):
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
        best_encrypted = fmin(fn=self.objective_function,
                              space=self.params, algo=tpe.suggest,
                              max_evals=iterations,
                              trials=hyperopt_trials,
                              timeout=timeout_sec)
        self.best_params = space_eval(space=self.params,
                                      hp_assignment=best_encrypted)
        return self.best_params, self.best_model

    def greater_is_better(self):
        """
        Extracting the private field of scorer to check
        whether the metric has a property of greater_is_better
        """
        is_greater = True if self.scorer._sign == 1 else False
        return is_greater


class FLOAdapter(TunerAdapter):
    def __init__(self, tuner: 'FLOTuner'):
        super(FLOAdapter, self).__init__(tuner)
        self.params = tuner.params_range

    def space_explore(self, base: int, complexity: int,
                      space_r: dict, space_nr: dict,
                      base_space: dict):
        """
        Propose two hyperparameters variations
        For complexity-related hyperparameters one increase while the other decrease complexity
        For complexity-not-related hyperparameters we change the value randomly.

        Parameters:
        base : The geometric base defines the scale of the change.
        complexity : 0 - for complexity-related hyperparameters.
                     1 - forcomplexity-not-related hyperparameters.
        space_r : Complexity-related hyperparameters Space.
        space_nr : Complexity-not-related hyperparameters Space.
        base_space : Base hyperparameters Space.

        Returns:
        positive (base_positive_variation)
        and negative (base_negative_variation) variation on the base hyperparameter.
        """

        base_positive_variation = deepcopy(base_space)
        base_negative_variation = deepcopy(base_space)
        if complexity == 0:
            unit_vector = self.unit_norm(space_r)

            base_positive_variation[1].update(base_space[1])
            base_negative_variation[1].update(base_space[1])

            for param in space_r.keys():
                # Updating the new hyperparameter values.
                base_positive_variation[complexity][param] = base_space[complexity][param] * base ** unit_vector[param]
                base_negative_variation[complexity][param] = base_space[complexity][param] * base ** -unit_vector[param]

                # Adjusting values that go out of range.
                if base_positive_variation[complexity][param] > space_r[param][1]:
                    base_positive_variation[complexity][param] = space_r[param][1]

                if base_negative_variation[complexity][param] < space_r[param][0]:
                    base_negative_variation[complexity][param] = space_r[param][0]

                # Hyperparameter that are larger than 1 should be integers.
                if space_r[param][1] > 1:
                    base_negative_variation[complexity][param] = int(
                        np.ceil(base_negative_variation[complexity][param]))
                    base_positive_variation[complexity][param] = int(
                        np.ceil(base_positive_variation[complexity][param]))
        else:
            # We set the complexity-related hyperparameters.
            base_positive_variation[0].update(base_space[0])
            base_negative_variation[0].update(base_space[0])

            # Random search for complexity-not-related hyperparameters.
            base_positive_variation[1] = {k: sample(space_nr[k], 1)[0] for k in space_nr.keys()}
            base_negative_variation[1] = {k: sample(space_nr[k], 1)[0] for k in space_nr.keys()}

        return base_positive_variation, base_negative_variation

    def unit_norm(self, space):
        """
        Create a length(S)-dimensional random unit vector u.

        Parameters:
        S : Complexity-related hyperparameters Space.

        Returns:
        u : random normalize unit vector
        """
        # Generating a random value for each component in the hyperparameters Space.
        vec = [np.random.uniform(0, 1, 1)[0] for _ in space.keys()]

        # Normalize to create a unit vector.
        mag = sum(x ** 2 for x in vec) ** .5
        u = {k: vec[ind] / mag for ind, k in enumerate(space.keys())}

        return u

    def objective_function(self, params, sample_size):

        group_kfold = GroupKFold(n_splits=10)
        groups = self.data.idx

        """
        Training the model and calculating the performance.

        Parameters :
        params : The model hyperparameters.
        sample_size : sample size.

        Returns:
        scores : list of model performance obtained by the cross validation.
        """
        scores = []

        # Cross-validation groups separated on data idx.
        for train_index, test_index in group_kfold.split(self.data.features, self.data.target, groups):
            X_train, X_test = self.data.features[train_index], self.data.features[test_index]
            y_train, y_test = self.data.target[train_index], self.data.target[test_index]

            # Training the model.
            sampled_features_train = X_train[0:sample_size, :]
            sampled_target_train = y_train[0:sample_size]
            self.model_to_adapt.set_params(**params)
            self.model_to_adapt.fit(sampled_features_train, sampled_target_train)

            # Aggregating the results of the iterations as a list.
            scores.append(abs(self.scorer(self.model_to_adapt, X_test, y_test)))
        return np.array(scores).mean(), scores

    def tune(self):
        space_r, space_nr = self.params.values()

        # Search space definition
        full_space = self._initialise_hyperparams_dict(space_r, space_nr)
        space_positive_direction = deepcopy(full_space)
        space_negative_direction = deepcopy(full_space)

        default_related_params = deepcopy(full_space[0])
        default_non_related_params = deepcopy(full_space[1])

        dimension = [len(space_r), len(space_nr)]

        # if complexity related parameters not empty
        if dimension[0] > 0:
            default_complexity = 0
            threshold = 2 ** (dimension[0] - 1)
        else:
            default_complexity = 1
            threshold = 2 ** (dimension[1] - 1)

        complexity = default_complexity

        # initial sample size
        default_sample_size = 100
        sample_size = default_sample_size

        # Geometric Base
        default_base_value = 2
        base = default_base_value

        # Number of times we fail to ﬁnd an improvement over the current conﬁguration
        failed_improvements_number = 0

        performance = list()

        # train default model
        start_time = timeit.default_timer()
        mean_score, scores = self.objective_function({**space_positive_direction[0],
                                                      **space_positive_direction[1]},
                                                     sample_size)

        end_time = timeit.default_timer()

        # Expected time
        expected_time = end_time - start_time
        expected_improvement_time = 2 * expected_time

        current_model_score, current_model_scores, performance, last_config_time, \
        second_to_last_config_time = self.verbose_state(end_time,
                                                        expected_improvement_time,
                                                        expected_time,
                                                        sample_size,
                                                        complexity,
                                                        failed_improvements_number,
                                                        base, mean_score,
                                                        mean_score,
                                                        scores, scores,
                                                        {**space_positive_direction[0],
                                                         **space_positive_direction[1]},
                                                        'BASE MODEL',
                                                        performance)

        for iteration in range(50):
            print(iteration)
            if expected_time <= expected_improvement_time:
                space_positive_direction, space_negative_dicrection = \
                    self.space_explore(base=base,
                                       complexity=complexity,
                                       space_r=space_r,
                                       space_nr=space_r,
                                       base_space=full_space)

                # EXPLORE POSITIVE
                # TRAIN the model on the positive direction
                score_positive, scores_positive = self.explore_direction(space_positive_direction,
                                                                         sample_size)

                # UPDATE
                if self.tuner.is_score_better(previous=current_model_score,
                                              current=score_positive):
                    """
                    Update the times for the latest two best conﬁgurations.
                    Report and document model performance 
                    Update Base model parameters and performance"""

                    current_model_score, current_model_scores, performance, last_config_time, \
                    second_to_last_config_time = self.verbose_state(last_config_time, expected_improvement_time,
                                                                    expected_time, sample_size,
                                                                    complexity,
                                                                    failed_improvements_number,
                                                                    base,
                                                                    current_model_score, score_positive,
                                                                    scores_positive, current_model_scores,
                                                                    {**space_positive_direction[0],
                                                                     **space_positive_direction[1]},
                                                                    'UPDATE C+', performance)
                    # Update the base model.
                    full_space[complexity].update(space_positive_direction[complexity])
                    expected_time = self._time_update(last_config_time, second_to_last_config_time)
                    self.best_params = {**space_positive_direction[0],
                                        **space_positive_direction[1]}
                    continue
                else:
                    print('No Improvment for C+')

                # EXPLORE NEGATIVE
                score_negative, scores_negative = self.explore_direction(space_negative_dicrection,
                                                                         sample_size)

                # UPDATE
                if self.tuner.is_score_better(previous=current_model_score,
                                              current=score_negative):

                    current_model_score, current_model_scores, performance, last_config_time, \
                    second_to_last_config_time = self.verbose_state(last_config_time, expected_improvement_time,
                                                                    expected_time, sample_size,
                                                                    complexity, failed_improvements_number,
                                                                    base, current_model_score, score_negative,
                                                                    scores_negative, current_model_scores,
                                                                    {**space_negative_direction[0],
                                                                     **space_negative_direction[1]},
                                                                    'UPDATE C-', performance)
                    # Update the base model.
                    full_space[complexity].update(space_negative_dicrection[complexity])
                    expected_time = self._time_update(last_config_time, second_to_last_config_time)
                    self.best_params = {**space_negative_dicrection[0],
                                        **space_negative_dicrection[1]}
                    continue
                else:
                    print('No Improvment for C-')

                # IF NO IMPROVEMENTS ON DIRECTIONS, WORK WITH COMPLEXITY

                complexity = 1 - complexity
                print('update complexity =', complexity)

                # If no complexity-not-related hyperparameters, switch back.
                if dimension[complexity] == 0:
                    complexity = 1 - complexity
                    print('update complexity =', complexity)

                if complexity == 1 or dimension[1] == 0:
                    # When no performance improves, update the counter.
                    failed_improvements_number += 1

                    print('update failed_improvements_number =',
                          failed_improvements_number)
                    # If the counter reaches the threshold.
                    # Tries to decrease the geometric base because
                    # model may be close to a local optimum.
                    if failed_improvements_number == threshold:
                        failed_improvements_number = 0
                        base = np.sqrt(base)
                        print('update base=', base)

                        expected_changes = [(1 + 1 / full_space[0][k]) ** np.sqrt(dimension[0])
                                            for k in space_r.keys()]
                        if base <= min(expected_changes):
                            base = min(base ** 2, max([(k[1] / k[0]) ** np.sqrt(dimension[0])
                                                       for k in space_r.values()]))
                            print('update base=', base)
                            # Update the Base parameters with default values.
                            if dimension[0] > 0:
                                full_space[0] = deepcopy(default_related_params)
                                print('update related params in base space=', full_space[0])
                            else:
                                full_space[1] = deepcopy(default_non_related_params)
                                print('update non related params in base space=', full_space[1])
                            self.best_params.update({**full_space[0], **full_space[1]})
                            complexity = default_complexity
                            sample_size = default_sample_size
                            print('Model reset, update to default values')
            else:
                sample_size *= 2
                if sample_size > len(self.data.target):
                    sample_size = len(self.data.target)

                start_time = timeit.default_timer()
                mean_score, scores = self.objective_function({**full_space[0],
                                                              **full_space[1]},
                                                             sample_size)

                end_time = timeit.default_timer()

                if self.tuner.is_score_better(previous=current_model_score,
                                              current=mean_score):
                    current_model_score, current_model_scores, performance, last_config_time_0, \
                    second_to_last_config_time_0 = self.verbose_state(end_time, expected_improvement_time,
                                                                    expected_time, sample_size,
                                                                    complexity, failed_improvements_number,
                                                                    base, current_model_score, mean_score,
                                                                    scores, current_model_scores,
                                                                    {**space_positive_direction[0],
                                                                     **space_positive_direction[1]},
                                                                    # {**full_space[0],
                                                                    #  **full_space[1]},
                                                                    'UPDATE SIZE', performance)
                    print('Performance Improve by UPDATE SIZE', "\r\n")
                    self.best_params = {**space_positive_direction[0],
                                        **space_positive_direction[1]}
                else:
                    print('No Improvment for UPDATE SIZE', "\r\n")

                expected_improvement_time = 2 * (end_time - start_time)

            expected_time = self._time_update(last_time=last_config_time,
                                              prelast_time=second_to_last_config_time)

        return self.best_params, self.best_model

    def explore_direction(self, vector, sample_size):
        mean_score, scores = self.objective_function({**vector[0],
                                                      **vector[1]},
                                                     sample_size)
        return mean_score, scores

    @staticmethod
    def _time_update(last_time, prelast_time):
        current_improvement_time = timeit.default_timer()
        expected_time = max(current_improvement_time - last_time,
                            last_time - prelast_time)
        return expected_time

    @staticmethod
    def _initialise_hyperparams_dict(space_r, space_nr):
        c = {0: {k: space_r[k][0] for k in space_r.keys()},
             1: {k: space_nr[k][0] for k in space_nr.keys()}}
        return c

    @staticmethod
    def verbose_state(last_config_time, expected_improvement_time,
                      expected_time, sample_size, complexity, failed_attempts,
                      base, cm, cmU, cmpU, cml,
                      params, text, performace):
        """
        Report and document the results

        Parameters:
        last_config_time : times for the latest best conﬁgurations
        expected_improvement_time :  expected time for improvement for the larger sample size.
        expected_time : expected time for improvement for the current sample size.
        sample_size : Sample Size.
        complexity : Complexity-related or Complexity-not-related hyperparameters.
        failed_attempts : the counter of trials we fail to ﬁnd an improvement.
        base : The geometric base for the change.
        cm : Avg Performance of the Last optimal model.
        cmU : Avg Performance of the current model.
        cmpU : list of all Performance of the current model.
        cml : list of all Performance of the Last optimal model.
        params : hyperparameters of the last model.
        performace : list of the performance results.
        text : just a text that include in the printout

        Returns:
        c: Avg Performance of the current model.
        cl: list of all Performance of the current model.
        performace : Updated list of the performance results.
        last_config_time, second_to_last_config_time : times for the latest two best conﬁgurations
        """
        # Update the times for the latest two best conﬁgurations.
        second_to_last_config_time = last_config_time
        last_config_time = timeit.default_timer()

        # Report and document model performance
        print(text)
        print('Et =', "%.2f" % expected_improvement_time, 'sec',
              ', E =', "%.2f" % expected_time, 'sec',
              ', Sample Size =', "%.2f" % sample_size,
              ', p =', complexity,
              ', n =', failed_attempts,
              ', B =', base
              )
        print('Base Performance =', "%.2f" % cm,
              ', UPDATE Performance =', "%.2f" % cmU,
              ', p-value=', "%.2f" % stats.ttest_ind(cmpU, cml).pvalue
              )
        print('Hyper-PARAM:', params, "\r\n")

        performace.append([text, expected_improvement_time,
                           expected_time, sample_size, complexity,
                           failed_attempts, base, cm, cmU, cmpU, cml, params])

        # Update Base model parameters and performance
        c = cmU
        cl = cmpU

        return c, cl, performace, last_config_time, second_to_last_config_time
