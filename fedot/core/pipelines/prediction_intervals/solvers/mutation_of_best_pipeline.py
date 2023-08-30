import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from golem.core.log import LoggerAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.data.data import InputData
from fedot.core.composer.metrics import RMSE
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast

from fedot.core.pipelines.prediction_intervals.ts_mutation import get_mutations, get_different_mutations
from fedot.core.pipelines.prediction_intervals.pipeline_constraints import first_prediction_constraint, \
    deviance_constraint


def solver_mutation_of_best_pipeline(train_input: InputData,
                                     ind: Individual,
                                     horizon: int,
                                     number_mutations: int,
                                     mutations_choice: str,
                                     n_jobs: int,
                                     show_progress: bool,
                                     forecast: int,
                                     operations: List[str],
                                     logger: LoggerAdapter,
                                     discard_inapropriate_pipelines: bool,
                                     keep_percentage: float):
    """This function realizes 'mutation_of_best_pipeline' method.

    Args:
        train_input: train time series
        ind: individual to make mutaions
        logger: prediction interval logger
        horizon: horizon to build forecast
        number_mutations: number mutations to use
        mutations_choice: choose mutations with ('with_replacement') or without replacement ('different')
        n_jobs: n_jobs
        show_progress: flag to show progress
        logger: prediction intervals logger
        discard_inapropriate_pipelines: flag to keep unreliable pipelines
        keep_percentage: percentage of mutations to keep regarding the RMSE-metric of their performance over train ts.

    Returns:
        a list of predictions to build prediction intervals.
    """

    if show_progress:
        logger.info('Creating mutations of final pipeline...')

    if mutations_choice == 'different':
        mutations_of_best_pipeline = get_different_mutations(individual=ind,
                                                             number_mutations=number_mutations,
                                                             operations=operations)
    elif mutations_choice == 'with_replacement':
        mutations_of_best_pipeline = get_mutations(individual=ind, 
                                                   number_mutations=number_mutations,
                                                   operations=operations)

    raw_predictions = []
    metric_values = []
    first_pred_constraints = []
    deviance_pred_constraints = []
    s = 1
    for p in mutations_of_best_pipeline:
        pipeline = PipelineAdapter().restore(p.graph)
        if show_progress:
            logger.info(f'Pipeline number {s}')
            s += 1
            pipeline.show()
            start_time = time.time()
        pipeline.fit(train_input)
        pred = out_of_sample_ts_forecast(pipeline=pipeline, input_data=train_input, horizon=horizon)
        metric_value = RMSE.get_value(pipeline=pipeline, reference_data=train_input, validation_blocks=2)
        if show_progress:
            end_time = time.time()
            logger.info(f'fitting time {end_time-start_time} sec')
            logger.info(f'RMSE-metric: {metric_value}')
            fig, ax = plt.subplots()
            ax.plot(range(len(pred)), pred)
        raw_predictions.append(pred)

        # for each mutation we compute its charactersitcs that used later on to eliminate aproiri bad pipelines
        if discard_inapropriate_pipelines:
            fpc = first_prediction_constraint(ts_train=train_input.features, forecast=forecast, prediction=pred)
            dc = deviance_constraint(ts_train=train_input.features, prediction=pred)
            metric_value = RMSE.get_value(pipeline=pipeline, reference_data=train_input, validation_blocks=2)
            metric_values.append(metric_value)
            first_pred_constraints.append(fpc)
            deviance_pred_constraints.append(dc)

    # here we remove apriori bad pipelines based on the following characteristics:
    # first_pred_constraints: dismiss pipelines with first forecasted value that are too far from the model_forecast
    # deviance_pred_constraints: dismiss pipelines, such their forecasts oscillate too much
    # we remove a specified percentage of pipelines with biggest RMSE metric.
    if discard_inapropriate_pipelines:
        predictions = []
        maximal_metric_value = np.quantile(np.array(metric_values), keep_percentage)
        for i, m in enumerate(mutations_of_best_pipeline):
            if first_pred_constraints[i] and deviance_pred_constraints[i] and metric_values[i] < maximal_metric_value:
                predictions.append(raw_predictions[i])
    else:
        predictions = raw_predictions

    if len(predictions) == 0:
        raise ValueError('Variable predictions is empty list. Try to increase number_mutations to solve this problem.')

    return predictions
