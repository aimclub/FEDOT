Time series forecasting example
==============================================


This example explains how to solve time series forecasting task using Fedot.

Generally Fedot provides a high-level API that enables you to use common fit/predict interface. To use API it is required
to import certain object:

.. code-block:: python

    from fedot import Fedot

Then we have to load data and split it on train and test set.
But firstly we need to define forecast length. Let it be equal 10.
We have to create ``InputData`` object - it contains features, indexes, and target (for time series forecasting task) the
target - time series itself. ``InputData.from_csv_time_series()`` may be used. Then we just split our time series into
train and test part (test part has 10 last time series elements).

.. code-block:: python

    from fedot.core.data.data import InputData
    from fedot.core.data.data_split import train_test_data_setup
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

    # specify the task and the forecast length (required depth of forecast)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=10))

    # load data from csv
    train_input = InputData.from_csv_time_series(task=task,
                                                 file_path='time_series.csv',
                                                 delimiter=',',
                                                 target_column='value')
    # split data for train and test
    train_data, test_data = train_test_data_setup(train_input)

.. note::

    There are 3 possible values for TaskType:
        * TaskTypesEnum.classification
        * TaskTypesEnum.regression
        * TaskTypesEnum.ts_forecasting


Initialize the FEDOT object and define the type of modeling problem. In this case, problem is ``ts_forecasting``.
Here we should pass Task.task_params to constructor because optimizer should know which forecasting length is.
We don't pass timeout parameter - that means optimization will stop only when it will approach plateau.
(But setting this parameter is strongly recommended by default)

.. code-block:: python

    # init model for the time-series forecasting
    model = Fedot(problem='ts_forecasting', task_params=task.task_params)

.. note::

    Class ``Fedot.__init__()`` has more parameters, e.g.
    ``n_jobs`` for parallelization. For more details, see the :doc:`FEDOT API </api/api>` section in our documentation.

Due to you work with time series data you only need to pass ``InputData`` object only.

.. code-block:: python

     # run AutoML model design
    best_pipeline = model.fit(train_data)

After the fitting is completed, you can look at the structure of the resulting pipeline.

In text format:

.. code-block:: python

    best_pipeline.print_structure()

And in plot format:

.. code-block:: python

    best_pipeline.show()

To obtain out-of-sample prediction for test data you need call ``forecast()`` method from ``Fedot`` class.
Out of sample means that your model will predict values based on historical values and it's own predictions
(if forecast length more than defined). This mode is more fair and clear.

.. code-block:: python

     # use model to obtain out-of-sample forecast with one step
    forecast = model.forecast(test_data)

The ``get_metrics()`` method estimates the quality of predictions according the selected metrics.

.. code-block:: python

     print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'])

Since you got a prediction and calculated metrics you can plot your prediction by calling ``plot_prediction()`` method.
For time series forecasting task it plots historical time series, real target values and prediction of the model.

.. code-block:: python

     model.plot_prediction()

You may interested to save the model. To perform that just call ``best_pipeline.save()``

.. code-block:: python

     best_pipeline.save(path='path_to_save_and_load', create_subdir=False, is_datetime_in_path=False)




To load fitted pipeline you can invoke ``.load()`` from just initialised ``Pipeline`` object method with passing path to your pipeline.

.. code-block:: python

     from fedot.core.pipelines.pipeline import Pipeline
     loaded_pipeline = Pipeline().load('path_to_save_and_load')

Also you should refit your model for a new data:

.. code-block:: python

     import pandas as pd
     from fedot.core.repository.dataset_types import DataTypesEnum
     new_data = InputData.from_csv_time_series(task=task,
                                                 file_path='new_time_series.csv',
                                                 delimiter=',',
                                                 target_column='value')
     loaded_pipeline.fit_from_scratch(new_data))

     forecast = loaded_pipeline.predict(forecast_length=10).predict # Note that we should take .predict field for prediction

.. note::

    For more detail about pipelines save and load, please visit `this section </basics/pipeline_save_load>`.

Thus by this example we learned how to solve time series forecasting task with Fedot.

