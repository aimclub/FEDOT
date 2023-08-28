from typing import Union, List

from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.log import LoggerAdapter
from golem.core.optimisers.opt_history_objects.individual import Individual

from fedot.core.data.data import InputData
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum

from fedot.core.pipelines.prediction_intervals.tuners import quantile_loss_tuners
from fedot.core.pipelines.prediction_intervals.utils import get_different_pipelines


def solver_last_generation_ql(train_input: InputData,
                              generation: List[Individual],
                              logger: LoggerAdapter,
                              show_progress: bool,
                              horizon: int,
                              number_models: Union[int, str],
                              nominal_error: float,
                              iterations: int,
                              minutes: float,
                              n_jobs: int,
                              up_tuner: SimultaneousTuner,
                              low_tuner: SimultaneousTuner):
    """This function realizes 'last_generation_ql' method.

    Args:
        train_input: train time series
        generation: last generation of Fedot-class object
        logger: prediction interval logger
        horizon: horizon to build forecast
        number_models: number_of_models; if 'max' then all models are used
        nominal_error: nominal error
        iterations: number iterations for default tuner
        minutes: number minutes for default tuner
        n_jobs: n_jobs for default tuner
        up_tuner, low_tuner: tuners that can be choosen instead of the default tuners.

    Returns:
        dictionary with lists consisting of np.arrays for building upper and lower prediction intervals.
    """
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))
    tuners = quantile_loss_tuners(up_quantile=1 - nominal_error / 2,
                                  low_quantile=nominal_error / 2,
                                  train_input=train_input,                         
                                  n_jobs=n_jobs,
                                  task=task,
                                  show_progress=show_progress,
                                  iterations=iterations,
                                  minutes=minutes)
    if up_tuner is None:
        up_tuner = tuners['up_tuner']
    if low_tuner is None:
        low_tuner = tuners['low_tuner']

    # take best pipelines from the last generation
    all_pipelines = get_different_pipelines(generation)
    number_avaliable_pipelines = len(all_pipelines)

    up_predictions = []
    low_predictions = []
    s = 1
    message = f'All {number_avaliable_pipelines} avaliable pipelines will be used for training.'

    if number_models == 'max':
        index_iterations = number_avaliable_pipelines
        if show_progress:
            logger.info(message)
    elif number_models > number_avaliable_pipelines:
        index_iterations = number_avaliable_pipelines
        if show_progress:
            logger.info('number_models > number of avaliable pipelines. ' + message)
    else:
        index_iterations = number_models
        if show_progress:
            logger.info(f'{number_models} best pipelines will be used for training.')

    for ind in all_pipelines[:index_iterations]:

        pipeline = PipelineAdapter().restore(ind.graph)
        if show_progress:
            logger.info(f'Fitting pipeline №{s}')
            s += 1
            pipeline.show()

        tuned_pipeline = up_tuner.tune(pipeline)
        tuned_pipeline.fit(train_input)
        preds = out_of_sample_ts_forecast(pipeline=tuned_pipeline, input_data=train_input, horizon=horizon)
        up_predictions.append(preds)

        tuned_pipeline = low_tuner.tune(pipeline)
        tuned_pipeline.fit(train_input)
        preds = out_of_sample_ts_forecast(pipeline=tuned_pipeline, input_data=train_input, horizon=horizon)
        low_predictions.append(preds)

    return {'up_predictions': up_predictions, 'low_predictions': low_predictions}
