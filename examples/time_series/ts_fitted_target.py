from matplotlib import pyplot as plt

from fedot.core.data.data import InputData
from fedot.core.pipelines.ts_wrappers import fitted_target
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from test.unit.pipelines.test_pipeline_ts_wrappers import get_simple_short_lagged_pipeline


def show_fitted_time_series(len_forecast=24):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    ts_input = InputData.from_csv_time_series(file_path='../../cases/data/time_series/metocean.csv',
                                              task=task, target_column='value')

    pipeline = get_simple_short_lagged_pipeline()
    train_predicted = pipeline.fit(ts_input)

    fitted_ts_10 = fitted_target(train_predicted, 10)
    fitted_ts_act = fitted_target(train_predicted)
    plt.plot(ts_input.idx, ts_input.target, label='Actual time series')
    plt.plot(fitted_ts_10.idx, fitted_ts_10.predict, label='Fitted values horizon 10')
    plt.plot(fitted_ts_act.idx, fitted_ts_act.predict, label='Fitted values all')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    show_fitted_time_series()
