How to compose the pipeline in an automated way
-----------------------------------------------

.. code:: python

   import pandas as pd

   # load all needed datasets
   dataset_to_train = pd.read_csv(...)
   dataset_to_validate = pd.read_csv(...)

   # specify additional training parameters
   problem, timeout, preset, metric_names, with_tuning, n_jobs, logging_level = ...
   target_col = ...  # 'target' by default

   # build model
   auto_model = Fedot(
      problem=problem, preset=preset, metric_names=metric_names, 
      with_tuning=with_tuning, n_jobs=n_jobs, loggging_level=logging_level,
      seed=42
   )

   # train model with the provided dataset
   pipeline = auto_model.fit(features=dataset_to_train, target=target_col)

   # get scores for the prediction
   prediction = auto_model.predict(features=dataset_to_validate)

   # get fitting metrics
   validation_target = dataset_to_validate[target_col]
   auto_metrics = auto_model.get_metrics(validation_target)
   print(f'metrics: {auto_metrics}')
   auto_model.plot_prediction()
