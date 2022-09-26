How to create your own composite model in a manual way
------------------------------------------------------

-  **Step 1**. Specify problem type and choose dataset.

.. code:: python

   import pandas as pd

   model = Fedot(problem='classification')

   dataset_to_train = pd.read_csv(train_file_path)
   dataset_to_validate = pd.read_csv(train_file_path)

-  **Step 2**. Create *Pipeline* instance, create nodes with desired models

.. code:: python

   node_first = PrimaryNode('logit')
   node_second= PrimaryNode('xgboost')
   node_final = SecondaryNode('knn', nodes_from=[node_first, node_second])
   pipeline = Pipeline(node_final)

-  **Step 3**. Fit the pipeline using *fit* method.

.. code:: python

   model.fit(features=dataset_to_train, predefined_model=pipeline)

-  **Step 4**. Obtain the prediction using *predict* method.

.. code:: python

   prediction = model.predict(features=dataset_to_validate)
