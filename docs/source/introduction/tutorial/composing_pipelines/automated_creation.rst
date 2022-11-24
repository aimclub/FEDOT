Automated way
-------------

-  **Step 1**. Specify problem type, create the FEDOT model and load the datasets.

.. code:: python

   import pandas as pd

   # specify additional training parameters
   timeout, with_tuning, n_jobs, logging_level = ...  # tested with 3, False, 1, logging.FATAL 

.. hint::

    See the :doc:`API documentation </api/api>` for descriptions of these and others parameters.

.. code:: python

   # build model
   auto_model = Fedot(
      problem='classification', timeout=timeout, preset='fast_train', 
      with_tuning=with_tuning, n_jobs=n_jobs, loggging_level=logging_level,
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

-  **Step 2**. Fit the model's pipeline.

.. code:: python

   pipeline = auto_model.fit(features=dataset_to_train, target=target_col)

-  **Step 3**. Obtain the prediction and calculate the metrics.

.. code:: python

   # get the prediction
   prediction = auto_model.predict(features=dataset_to_validate)

   # calculate the scores
   auto_metrics = auto_model.get_metrics(validation_target)
   print(f'metrics: {auto_metrics}')
   >>> metrics: {'roc_auc': 0.833, 'f1': 0.936}

Congratulations! We've just automatically built a machine learning pipeline with FEDOT.
This time our pipeline outperforms the assumptions from the :doc:`previous example <manual_creation>`.
To continue exploring the framework, there are pages with more detailed information on using FEDOT:

-  :doc:`/basics/index`;
-  :doc:`/examples/index`;
-  :doc:`/api/index`;
-  :doc:`/advanced/index`.
