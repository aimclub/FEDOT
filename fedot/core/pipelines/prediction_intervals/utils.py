import numpy as np
from typing import List

from fedot.core.pipelines.ts_wrappers import fitted_values
from fedot.core.pipelines.pipeline import Pipeline
from fedot import Fedot
from fedot.core.data.data import InputData
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot.core.pipelines.prediction_intervals.params import PredictionIntervalsParams
from fedot.core.pipelines.prediction_intervals.graph_distance import get_distance_between


def compute_prediction_intervals(arrays: List[np.array], nominal_error: int = 0.1):
    """Provided a list of np.arrays this function computes upper and low quantiles, max, min, median and mean arrays."""

    arrays = np.array(arrays)

    return {'quantile_up': np.quantile(arrays, q=1 - nominal_error / 2, axis=0, method='median_unbiased'),
            'quantile_low': np.quantile(arrays, q=nominal_error / 2, axis=0, method='median_unbiased'),
            'mean': np.mean(arrays, axis=0),
            'median': np.median(arrays, axis=0),
            'max': np.max(arrays, axis=0),
            'min': np.min(arrays, axis=0)}


def get_different_pipelines(individuals: List[Individual]):
    """This function removes individuals with identical graphs in a given list of individuals.

    Args:
        individuals: list of Individual objects.

    Returns:
        list of Individuals with different graphs. Given individuals with identical graph individual
        with better (smaller) fitness is taken.
    """
    sorted_inds = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    graph_list = []
    new_inds = []
    for s in sorted_inds:
        if np.array([get_distance_between(s.graph, x, compare_node_params=False) > 0 for x in graph_list]).all():
            graph_list.append(s.graph)
            new_inds.append(s)
    return new_inds


def get_last_generations(model: Fedot):
    """This function takes final individual and last generation from fitted Fedot-class object."""

    # TODO: in the future we should take here a copy. The problem is that Individual object is not copied.
    generations = model.history.generations
    if len(generations) < 2:
        raise ValueError('Model has < 2 generations. Please fit model and try again.')
    sorted_final_generation = sorted(generations[-1], key=lambda x: x.fitness.value)
    return {'final_choice': sorted_final_generation[0], 'last_generation': generations[-2]}


def get_base_quantiles(train_input: InputData, pipeline: Pipeline, nominal_error: int):
    """This function computes quantiles of residuals of forecast of a given pipeline over train data."""

    p = pipeline.fit(train_input)
    preds = fitted_values(train_input, p).predict
    train = train_input.features
    resids = train[-len(preds):] - preds

    ans = {'low': np.quantile(resids, nominal_error / 2, method='median_unbiased'),
           'up': np.quantile(resids, 1 - nominal_error / 2, method='median_unbiased')}

    return ans


def ts_jumps(ts: np.array):
    """This function computes maximal upswing and downswing of a time series."""

    ts_diff = np.diff(ts)
    return {'up': np.max(ts_diff), 'low': np.min(ts_diff)}


def ts_deviance(ts: np.array):
    """This function computes average module of difference between neighboring elements of time series."""

    return np.mean(np.abs(np.diff(ts)))


def check_init_params(model: Fedot,
                      horizon: int,
                      nominal_error: float,
                      method: str,
                      params: PredictionIntervalsParams):
    """This function checks correctness of parameters initializing PredictionInterval instance."""

    if not model.current_pipeline:
        raise ValueError('Fedot class object is not fitted.')

    if horizon is not None:
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError('Argument horizon must be None or positive integer number.')

    if type(nominal_error) not in [int, float] or not (0 <= nominal_error <= 1):
        raise ValueError('Argument nominal_error must be float number between 0 and 1.')

    avaliable_methods = ['last_generation_ql', 'best_pipelines_quantiles', 'mutation_of_best_pipeline']
    if method not in avaliable_methods:
        raise ValueError('''Argument 'method' is incorrect. Possible options: 'last_generation_ql',
'best_pipelines_quantiles', 'mutation_of_best_pipeline'.''')

    if params.logging_level not in [0, 10, 20, 30, 40, 50]:
        raise ValueError('Argument logging_level must be in [0, 10, 20, 30, 40, 50].')

    if not isinstance(params.n_jobs, int) or params.n_jobs == 0 or params.n_jobs < -1:
        raise ValueError('Argument n_jobs must be -1 or positive integer number.')

    if not isinstance(params.show_progress, bool):
        raise ValueError('Argument show_progress must be boolean.')

    if not isinstance(params.number_mutations, int) or params.number_mutations < 1:
        raise ValueError('Argument number_mutations must be positive integer number.')

    if params.mutations_choice not in ['different', 'with_replacement']:
        raise ValueError("Arument mutations_choice is incorrect. Options: 'different' and 'with_replacement'.")

    if not isinstance(params.mutations_discard_inapropriate_pipelines, bool):
        raise ValueError('Argument mutations_discard_inapropriate_pipelines must be boolean.')

    if type(params.mutations_keep_percentage) not in [int, float] or not (0 <= params.mutations_keep_percentage <= 1):
        raise ValueError('Argument mutation_keep_percentage must be float number between 0 and 1.')

    if not isinstance(params.mutations_operations, list):
        raise ValueError('Argument mutations_operations must be list of strings.')

    if params.ql_number_models != 'max':
        if not isinstance(params.ql_number_models, int) or params.ql_number_models < 1:
            raise ValueError("Argument ql_number_models must be positive integer number or 'max'.")

    if not isinstance(params.ql_tuner_iterations, int) or params.ql_tuner_iterations < 1:
        raise ValueError('Argument ql_tuner_iterations must be positive integer number.')

    if type(params.ql_tuner_minutes) not in [int, float] or params.ql_tuner_minutes <= 0:
        raise ValueError('Argument ql_tuner_minutes must be positive real number.')

    if params.bpq_number_models != 'max':
        if not isinstance(params.bpq_number_models, int) or params.bpq_number_models < 1:
            raise ValueError("Argument bpq_number_models must be positive integer number or 'max'.")
