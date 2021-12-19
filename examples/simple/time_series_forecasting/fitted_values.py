from matplotlib import pyplot as plt

from fedot.core.data.data import InputData
from fedot.core.pipelines.ts_wrappers import fitted_values, in_sample_fitted_values
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from test.unit.pipelines.test_pipeline_ts_wrappers import get_simple_short_lagged_pipeline


def show_fitted_time_series(len_forecast=24):
    """
    Shows an example of how to get fitted values of a time series by any
    pipeline created by FEDOT

    fitted values - are the predictions of the pipelines on the training sample.
    For time series, these values show how well the model reproduces the time
    series structure
    """
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    ts_input = InputData.from_csv_time_series(file_path='../../../cases/data/time_series/metocean.csv',
                                              task=task, target_column='value')

    pipeline = get_simple_short_lagged_pipeline()
    train_predicted = pipeline.fit(ts_input)

    # Get fitted values for every 10th forecast
    fitted_ts_10 = fitted_values(ts_input, train_predicted, 10)
    # Average for all forecasting horizons
    fitted_ts_act = fitted_values(ts_input, train_predicted)
    # In-sample forecasting fitted values
    in_sample_validated = in_sample_fitted_values(ts_input, train_predicted)

    plt.plot(range(len(ts_input.idx)), ts_input.target, label='Actual time series', alpha=0.8)
    plt.plot(fitted_ts_10.idx, fitted_ts_10.predict, label='Fitted values horizon 10', alpha=0.2)
    plt.plot(fitted_ts_act.idx, fitted_ts_act.predict, label='Fitted values all', alpha=0.2)
    plt.plot(in_sample_validated.idx, in_sample_validated.predict, label='In-sample fitted values')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    show_fitted_time_series()
