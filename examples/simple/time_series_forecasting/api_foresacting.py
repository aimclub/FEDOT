import pandas as pd

from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.utils import fedot_project_root


def run_ts_forecasting_example(with_plot=True, with_pipeline_vis=True, timeout: float = None):
    train_data_path = f'{fedot_project_root()}/examples/data/ts/salaries.csv'

    target = pd.read_csv(train_data_path)['value']

    # Define forecast length and define parameters - forecast length
    forecast_length = 30
    task_parameters = TsForecastingParams(forecast_length=forecast_length)

    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting', task_params=task_parameters, timeout=timeout,
                  preset='fast_train')

    # run AutoML model design in the same way
    pipeline = model.fit(features=train_data_path, target='value')
    if with_pipeline_vis:
        pipeline.show()

    # use model to obtain forecast
    forecast = model.predict(features=train_data_path)

    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))

    # plot forecasting result
    if with_plot:
        model.plot_prediction()

    return forecast


if __name__ == '__main__':
    run_ts_forecasting_example(timeout=0.5)
