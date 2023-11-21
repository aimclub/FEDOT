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
                              task=Task(TaskTypesEnum.classification)) # or regression
    train, test = train_test_data_setup(data)

.. note::

    There are 3 possible values for TaskType:
    * TaskTypesEnum.classification
    * TaskTypesEnum.regression
    * TaskTypesEnum.ts_forecasting

You also can create InputData from pandas `DataFrame`:

.. code-block:: python


    from fedot.core.data.data import InputData


    data = InputData.from_dataframe(features_df,
                                    target_df,
                                    task=Task(TaskTypesEnum.classification)) # or regression
    train, test = train_test_data_setup(data)

or from numpy array:

.. code-block:: python


    from fedot.core.data.data import InputData


    data = InputData.from_numpy(features_array,
                                target_array,
                                task=Task(TaskTypesEnum.classification)) # or regression
    train, test = train_test_data_setup(data)

After you can split data on train/test set:

.. code-block:: python


    train, test = train_test_data_setup(data)

and pass it to model:

.. code-block:: python


    model = Fedot(...)
    model.fit(train)
    model.predict(test)

For time series forecasting problem
