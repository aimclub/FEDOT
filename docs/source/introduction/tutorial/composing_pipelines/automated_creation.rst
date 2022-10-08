Automated way
-------------

-  **Step 1**. Specify problem type, load the model and datasets.

.. code:: python

   import pandas as pd

   # specify additional training parameters
   timeout, with_tuning, n_jobs, logging_level = ...  # tested with 3, False, 1, logging.FATAL 

you can look the meaning of that and other parameters, see :class:`~fedot.api.main.Fedot`

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

   # concretise target column and validation answers data
   target_col: str = ...  # 'target' by default
   validation_target = dataset_to_validate[target_col]

-  **Step 2**. Fit the model's pipeline using ``fit`` method.

.. code:: python

   # train model with the provided dataset
   pipeline = auto_model.fit(features=dataset_to_train, target=target_col)

-  **Step 3**. Obtain the prediction using ``predict`` method and calculate the chosen metrics.

.. code:: python

   # get scores for the prediction
   prediction = auto_model.predict(features=dataset_to_validate)

   # calculate the chosen metrics
   auto_metrics = auto_model.get_metrics(validation_target)
   print(f'metrics: {auto_metrics}')
   >>> metrics: {'roc_auc': 0.833, 'f1': 0.936}
   auto_model.plot_prediction()
