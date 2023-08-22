import numpy as np
import pickle
from typing import List

from fedot.core.pipelines.ts_wrappers import fitted_values
from fedot.core.pipelines.pipeline import Pipeline
from golem.core.optimisers.opt_history_objects.individual import Individual
from fedot.api.main import Fedot
from fedot.core.data.data import InputData


def compute_prediction_intervals(arrays: List[np.array], nominal_error: int = 0.1):
    """Provided a list of np.arrays this function computes upper and low quantiles, max, min, median and mean arrays."""

    arrays = np.array(arrays)
    number_arrays = arrays.shape[0]
    len_array = arrays.shape[1]
    arrays = arrays.flatten().reshape((number_arrays, len_array))

    return {'quantile_up': np.quantile(arrays, q=1 - nominal_error / 2, axis=0, method='median_unbiased'),
            'quantile_low': np.quantile(arrays, q=nominal_error / 2, axis=0, method='median_unbiased'),
            'mean': np.mean(arrays, axis=0),
            'median': np.median(arrays, axis=0),
            'max': np.max(arrays, axis=0),
            'min': np.min(arrays, axis=0)}


def pipeline_simple_structure(ind: Individual):
    """This function transformates an individual to a list consisting of its pipeline node names.

    Args:
        ind: some pipeline.

    Returns:
        list of node names in string type.
    """
    return list(map(lambda x: x.name, ind.graph.graph_description['nodes']))


def get_different_pipelines(individuals: List[Individual]):
    """This function removes individuals with identical pipeline structure in a given list of individuals.

    Args:
        individuals: list of Individual type objects.

    Returns:
        list of Individual type objects with different pipeline structures.
    """
    unique_inds = []
    for ind in individuals:
        if ind not in unique_inds:
            unique_inds.append(ind)
    l_inds = [(ind.fitness.value, pipeline_simple_structure(ind), ind.uid) for ind in unique_inds]
    l_inds = sorted(l_inds, key=lambda x: x[0])
    structures = []
    ids = []
    for x in l_inds:
        if x[1] not in structures:
            structures.append(x[1])
            ids.append(x[2])

    return [ind for ind in unique_inds if ind.uid in ids]


def model_copy(model: Fedot, file_name='model_copy.pickle'):
    """This function copies Fedot class object."""

    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
    with open(file_name, 'rb') as f:
        return (pickle.load(f))


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


def check_init_params(model: Fedot, method: str):
    """This function checks correctness of parameters needed to initialize PredictionInterval instance."""

    avaliable_methods = ['last_generation_ql', 'best_pipelines_quantiles', 'mutation_of_best_pipeline']

    if method not in avaliable_methods:
        raise ValueError('''Argument 'method' is incorrect. Possible options: 'last_generation_ql',
'best_pipelines_quantiles', 'mutation_of_best_pipeline'.''')
    if not model.current_pipeline:
        raise ValueError('Fedot class object is not fitted.')
