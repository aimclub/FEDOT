import numpy as np
from matplotlib import pyplot as plt
from fedot.api.main import Fedot
from examples.advanced.time_series_forecasting.multi_ts_arctic_forecasting import prepare_data, initial_pipeline


def run_multi_ts_forecast(forecast_length, multi_ts):
    train_data, test_data, task = prepare_data(forecast_length, multi_ts)

    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=10,
                  initial_assumption=initial_pipeline(),
                  composer_params={
                      'max_depth': 4,
                      'num_of_generations': 20,
                      'timeout': 10,
                      'pop_size': 10,
                      'max_arity': 3,
                      'cv_folds': None,
                      'available_operations': ['lagged', 'smoothing', 'diff_filter', 'gaussian_filter',
                                               'ridge', 'lasso', 'linear', 'cut']
                  })

    # fit model
    pipeline = model.fit(train_data)
    pipeline.show()
    # use model to obtain forecast
    forecast = model.predict(test_data)
    target = np.ravel(test_data.target)

    # visualize results
    if multi_ts:
        history = np.ravel(train_data.target[:, 0])
    else:
        history = np.ravel(train_data.target)
    plt.plot(np.ravel(test_data.idx), np.ravel(test_data.target), label='test')
    plt.plot(np.ravel(train_data.idx), history, label='history')
    plt.plot(np.ravel(test_data.idx), forecast, label='prediction_after_tuning')
    plt.xlabel('Time step')
    plt.ylabel('Sea level')
    plt.legend()
    plt.show()

    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))


if __name__ == '__main__':
    forecast_length = 60
    run_multi_ts_forecast(forecast_length, multi_ts=True)
    run_multi_ts_forecast(forecast_length, multi_ts=False)
