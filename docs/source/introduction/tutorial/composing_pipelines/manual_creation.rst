Manual way
----------

-  **Step 1**. Specify problem type, create the FEDOT model and load the datasets.

.. code:: python

   import pandas as pd

   # specify additional automl training parameters
   timeout, n_jobs, logging_level = ...  # tested with 3., 1, logging.FATAL

.. hint::

    See the :doc:`API documentation </api/api>` for descriptions of these and others parameters.

.. code:: python

   # build model for adjusting the pipeline
   model = Fedot(
      problem='classification', timeout=timeout, n_jobs=n_jobs, logging_level=logging_level,
      seed=42
   )

   # add all datasets paths and load datasets
   train_file_path: Union[str, os.Pathlike] = ...
   validation_file_path: Union[str, os.Pathlike] = ...
   # tested with default scoring classification from FEDOT's datasets

   dataset_to_train = pd.read_csv(train_file_path)
   dataset_to_validate = pd.read_csv(validation_file_path)

   # specify target column and validation data
   target_col: str = ...  # 'target' by default
   validation_target = dataset_to_validate[target_col]

-  **Step 2**. Create *Pipeline* instance, i.e. create nodes with desired models.

.. code:: python

   node_first = PrimaryNode('logit')
   node_second = PrimaryNode('xgboost')
   node_final = SecondaryNode('knn', nodes_from=[node_first, node_second])
   pipeline = Pipeline(node_final)

The following pages provide examples of manually constructed pipelines for different tasks:
:doc:`classification </examples/classification_pipelines>`,
:doc:`regression </examples/regression_pipelines>`,
:doc:`time series forecasting </examples/ts_pipelines>`.

-  **Step 3**. Fit the pipeline.

.. code:: python

   model.fit(features=dataset_to_train, target=target_col, predefined_model=pipeline)

.. hint::

    If `predefined_model` is set to `'auto'`, FEDOT will choose and fit the default initial
    assumption for the task. Note that this differs from :doc:`automated model creation </introduction/tutorial/composing_pipelines/automated_creation>`,
    although it does not require manual selection of the model.

.. code::python

-  **Step 4**. Obtain a prediction and calculate the metrics.

.. code:: python

   # get the prediction
   prediction = model.predict(features=dataset_to_validate)

   # calculate the scores
   metrics = model.get_metrics(validation_target)
   print(f'metrics: {metrics}')
   >>> metrics: {'roc_auc': 0.617, 'f1': 0.9205}

Eventually, we get a configured machine learning pipeline and its data-based predictions.
Let's say we're looking for a baseline against which we can compare the result.
We repeat the steps 3 and 4, using `predefined_model='auto'`, and get the metric values:

``'roc_auc': 0.785, 'f1': 0.934``

In this case, our manually constructed pipeline outperforms the FEDOT's first guess in terms of F1,
but significantly looses in ROC AUC. Apparently, there is a better way to go.

Next, we can try to create another ML pipeline by hand and see if it gives higher scores.
Or we can let FEDOT do it for us using evolutionary search.
See :doc:`/introduction/tutorial/composing_pipelines/automated_creation` for details.
