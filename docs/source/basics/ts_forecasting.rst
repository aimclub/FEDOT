Time Series Forecasting
=======================

FEDOT allows you to automate machine learning pipeline design for time series forecasting.
To extract features FEDOT uses lagged transformation (windowing method) which allows to represent time-series as
trajectory matrix. Therefore not only specific models for time series forecasting (such as
ARIMA and AR) can be used but also any machine learning method (knn, decision tree, etc.).
Time-series specific preprocessing methods,
like moving average smoothing or Gaussian smoothing are used as well.

.. image:: img/windowing_method.jpg
   :width: 45%

Simple example
~~~~~~~~~~~~~~

.. code-block:: python

    validation_blocks = 2

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=4))

    train_input = InputData.from_csv_time_series(task=task,
                                                 file_path='time_series.csv',
                                                 delimiter=',',
                                                 target_column = 'value')

    train_data, test_data = train_test_data_setup(train_input,
                                                  validation_blocks=validation_blocks)

    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=10,
                  n_jobs=-1,
                  cv_folds=2,
                  validation_blocks=validation_blocks,
                  preset='fast_train')

    # run AutoML model design
    pipeline = model.fit(train_data)
    pipeline.show()

    # use model to obtain forecast
    forecast = model.predict(test_data)
    target = np.ravel(test_data.target)
    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))

    # plot forecasting result
    model.plot_prediction()

Time-series validation
~~~~~~~~~~~~~~~~~~~~~~

You can set

