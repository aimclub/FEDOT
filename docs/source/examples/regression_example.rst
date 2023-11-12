Regression example
==============================================


This example explains how to solve regression task using Fedot.

Generally Fedot provides a high-level API that enables you to use common fit/predict interface. To use API it is required
to import certain object:

.. code-block:: python

    from fedot import Fedot

Then we have to load data and split it on train and test set.
Fedot uses it's own data object notation (``InputData``). It contains index,
features and target for each sample. You can create it from file using ``InputData.from_csv()`` method.
You need to provide ``Task`` object with type of task you want to solve. After that just split your data into train and test.

.. code-block:: python

    from fedot.core.data.data import InputData
    from fedot.core.data.data_split import train_test_data_setup
    from fedot.core.repository.tasks import Task, TaskTypesEnum
    data_path = 'path_to_data'

    data = InputData.from_csv(data_path,
                              task=Task(TaskTypesEnum.regression))
    train, test = train_test_data_setup(data)

.. note::

    There are 3 possible values for TaskType:
        * TaskTypesEnum.classification
        * TaskTypesEnum.regression
        * TaskTypesEnum.ts_forecasting


Initialize the FEDOT object and define the type of modeling problem. In this case, problem is ``regression``.
You also can define seed parameter for reproducibility, timeout in minutes (in this example we limit fedot for 5 minutes).

.. code-block:: python

    model = Fedot(problem='regression', seed=42, timeout=5)

.. note::

    Class ``Fedot.__init__()`` has more parameters, e.g.
    ``n_jobs`` for parallelization. For more details, see the :doc:`FEDOT API </api/api>` section in our documentation.

To train our model we should call method ``fit()``. You need to provide ``train`` object to ``features`` field
and pass name of target column to ``target`` field. This method returns the best pipeline was obtained during optimization.

.. code-block:: python

    best_pipeline = model.fit(features=train, target='target')

After the fitting is completed, you can look at the structure of the resulting pipeline.

In text format:

.. code-block:: python

    best_pipeline.print_structure()

And in plot format:

.. code-block:: python

    best_pipeline.show()

To obtain prediction for test data you need call ``predict()`` method from ``Fedot`` class.

.. code-block:: python

    prediction = model.predict(features=test) # csv file should contains target column too for metric calculation

The ``get_metrics()`` method estimates the quality of predictions according the selected metrics.

.. code-block:: python

     print(model.get_metrics(rounding_order=4))  # we can control the rounding of metrics

.. note::

   You may see, that get_metrics() returned only RMSE metric. You can pass names of interested metrics by
   metrics_name parameter. F.e. ``get_metrics(metric_names=['mae', 'mse'])``.

Since you got a prediction and calculated metrics you can plot your prediction by calling ``plot_prediction()`` method.
For regression task it plots bi-plot.

.. code-block:: python

     model.plot_prediction()

You may interested to save the model. To perform that just call ``best_pipeline.save()``

.. code-block:: python

     best_pipeline.save(path='path_to_save_and_load', create_subdir=False, is_datetime_in_path=False)




To load fitted pipeline you can invoke ``.load()`` from just initialised ``Pipeline`` object method with passing path to your pipeline.

.. code-block:: python

     from fedot.core.pipelines.pipeline import Pipeline
     loaded_pipeline = Pipeline().load('path_to_save_and_load')

And you can do inference:

.. code-block:: python

     import pandas as pd
     from fedot.core.repository.dataset_types import DataTypesEnum
     new_features = pd.read_csv('new_data.csv')
     # note that we have to use fedot specific data type for inference for pipelines
     new_data_to_predict = InputData(features=new_features.values,
                                     target=None,  # if you don't know your target
                                     idx=new_features.index.values,
                                     task=Task(TaskTypesEnum.regression),
                                     data_type=DataTypesEnum.table)
     prediction = loaded_pipeline.predict(new_data_to_predict).predict # Note that we should take .predict field for prediction

Thus by this example we learned how to solve regression task with Fedot.

