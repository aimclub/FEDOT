How to compose the pipeline in an automated way
-----------------------------------------------

.. code:: python

   auto_model = Fedot(problem='classification')
   pipeline = auto_model.fit(features=dataset_to_train)
   prediction = auto_model.predict(features=dataset_to_validate)
   auto_metrics = auto_model.get_metrics()
