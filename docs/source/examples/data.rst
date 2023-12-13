Working with data
==============================================


This example explains how to load your data.

Fedot provides specific interface for operation with data.
Fedot uses it's own data object notation (InputData). It contains index,
features and target for each sample. You can create it from file using ``InputData.from_csv()`` method.
You need to provide ``Task`` object with type of task you want to solve.
Here examples of for tabular data:

.. code-block:: python


    from fedot.core.data.data import InputData
    data_path = 'path_to_data'

    data = InputData.from_csv(data_path,
    target_columns='target',
    task=Task(TaskTypesEnum.classification)) # or regression

.. note::

    There are 3 possible values for TaskType:
    * ``TaskTypesEnum.classification``
    * ``TaskTypesEnum.regression``
    * ``TaskTypesEnum.ts_forecasting``

.. note::

    You can provide several target columns (For regression task).Then Fedot will recognise it as multiregression task supported natively.

You also can create ``InputData`` from pandas ``DataFrame``:

.. code-block:: python


    from fedot.core.data.data import InputData


    data = InputData.from_dataframe(features_df,
                                    target_df,
                                    task=Task(TaskTypesEnum.classification)) # or regression

or from numpy array:

.. code-block:: python


    from fedot.core.data.data import InputData


    data = InputData.from_numpy(features_array,
                                target_array,
                                task=Task(TaskTypesEnum.classification)) # or regression

After you can split data on train/test set:

.. code-block:: python


    train, test = train_test_data_setup(data)

and pass it to the model:

.. code-block:: python


    model = Fedot(...)
    model.fit(train)
    model.predict(test)

For time series forecasting problem there is a little bit different approach for data initialization.
Firstly you need to create a ``Task`` object:

.. code-block:: python


    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
    # specify the task and the forecast length (required depth of forecast)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=your_forecast_length))



After that you can use ``Input_data.from_csv_series``

.. code-block:: python


    train_input = InputData.from_csv_time_series(task=task,
                                                 file_path='time_series.csv',
                                                 delimiter=',',
                                                 target_column='value')

But you also can create ``InputData`` from numpy :

.. code-block:: python


    train_input = InputData.from_numpy_time_series(series,
                                                   task=task)

After you can split data on train/test set (test set will contain last N values of the series by default):

.. code-block:: python


    train, test = train_test_data_setup(data)

and pass it to the model:

.. code-block:: python


    model = Fedot(...)
    model.fit(train)
    model.forecast()

Thus, this example shows how to operate with data in Fedot.