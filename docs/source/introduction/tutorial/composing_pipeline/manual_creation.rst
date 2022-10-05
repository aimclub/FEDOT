Manual way
----------

-  **Step 1**. Specify problem type and choose dataset.

.. code:: python

   import pandas as pd

   # specify additional automl training parameters
   timeout, preset, metric_names, with_tuning, n_jobs, logging_level = ...

   # build model for adjusting your own composite solution
   model = Fedot(
      problem='classification', timeout=timeout, metric_names=metric_names,
      with_tuning=with_tuning, n_jobs=n_jobs, loggging_level=loggging_level
   )

   # add all datasets paths and load datasets
   train_file_path = ...
   validation_file_path = ...

   dataset_to_train = pd.read_csv(train_file_path)
   dataset_to_validate = pd.read_csv(validation_file_path)

   # concretise target column and validation answers data
   target_col = ...
   validation_target = dataset_to_validate[target_col]

-  **Step 2**. Create *Pipeline* instance, create nodes with desired models

.. code:: python

   node_first = PrimaryNode('logit')
   node_second= PrimaryNode('xgboost')
   node_final = SecondaryNode('knn', nodes_from=[node_first, node_second])
   pipeline = Pipeline(node_final)

-  **Step 3**. Fit the chosen pipeline using *fit* method.

.. code:: python

   model.fit(features=dataset_to_train, predefined_model=pipeline)

-  **Step 4**. Obtain the prediction using *predict* method and show fitting metrics.

.. code:: python

   prediction = model.predict(features=dataset_to_validate)
   metrics = model.get_metrics(validation_target)
   print(f'metrics: {metrics}')
   model.plot_prediction()
