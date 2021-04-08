import numpy as np
import pandas as pd

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root


def run_classification_example():
    train_data_path = f'{project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    baseline_model = Fedot(problem=problem)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='xgboost')

    baseline_model.predict_proba(features=test_data_path)
    print(baseline_model.get_metrics())

    auto_model = Fedot(problem=problem, seed=42)
    auto_model.fit(features=train_data_path, target='target')
    prediction = auto_model.predict_proba(features=test_data_path)
    print(auto_model.get_metrics())

    return prediction


def prepare_input_for_forecasting(time_series, forecast_length):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    input = InputData(idx=np.arange(0, len(time_series)),
                      features=time_series,
                      target=time_series,
                      task=task,
                      data_type=DataTypesEnum.ts)
    train_input, predict_input = train_test_data_setup(input)

    return train_input, predict_input


def run_ts_forecasting_example(with_plot=True):
    train_data_path = f'{project_root()}/notebooks/jupyter_media/intro/salaries.csv'
    forecast_length = 30

    df = pd.read_csv(train_data_path)
    time_series = np.array(df['target'])

    train_data, target_data = prepare_input_for_forecasting(time_series, forecast_length)
    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting')

    # run AutoML model design in the same way
    actual_series = np.array(target_data.target)
    chain = model.fit(features=train_data, target=actual_series)
    chain.show()

    # use model to obtain forecast
    forecast = model.forecast(pre_history=train_data,
                              forecast_length=forecast_length)

    # plot forecasting result
    if with_plot:
        model.plot_prediction()

    return forecast


if __name__ == '__main__':
    run_classification_example()

    run_ts_forecasting_example()
