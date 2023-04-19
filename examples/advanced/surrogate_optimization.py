from functools import partial
from typing import Any

from golem.core.optimisers.meta.surrogate_model import SurrogateModel
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer

from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


class GraphDepthSurrogateModel(SurrogateModel):
    def __call__(self, graph, **kwargs: Any):
        # example how we can get input data from objective
        input_data = kwargs.get('objective').__self__.input_data
        return [len(graph.nodes)]


def run_ts_forecasting_example(dataset='australia', horizon: int = 30, validation_blocks=2, timeout: float = None,
                               visualization=False, with_tuning=True):
    train_data, test_data = get_ts_data(dataset, horizon, validation_blocks)
    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting',
                  task_params=Task(TaskTypesEnum.ts_forecasting,
                                   TsForecastingParams(forecast_length=horizon)).task_params,
                  timeout=timeout,
                  n_jobs=-1,
                  with_tuning=with_tuning,
                  cv_folds=2, validation_blocks=validation_blocks, preset='fast_train',
                  optimizer=partial(SurrogateEachNgenOptimizer, surrogate_model=GraphDepthSurrogateModel()))

    # run AutoML model design in the same way
    pipeline = model.fit(train_data)

    # use model to obtain two-step in-sample forecast
    in_sample_forecast = model.predict(test_data)
    print('Metrics for two-step in-sample forecast: ',
          model.get_metrics(metric_names=['rmse', 'mae', 'mape']))

    # plot forecasting result
    if visualization:
        pipeline.show()
        model.plot_prediction()


if __name__ == '__main__':
    run_ts_forecasting_example(visualization=True, timeout=3)
