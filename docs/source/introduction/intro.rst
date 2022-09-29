Intro to FEDOT
==============

FEDOT - an open-source framework for automated modeling and machine learning (AutoML). It produces a lightweight end-to-end solution in an automated way using an evolutionary approach.
.. |FEDOT logo| image:: img_intro/fedot_logo.png
   :width: 80%
FEDOT supports classification (binary and multiclass), regression, clustering, and time series forecasting tasks. FEDOT works both on unimodal (only tabular/image/text data) and multimodal data (more than one data source).
.. |Pipeline schema| image:: img_intro/small_pipeline.png
   :width: 80%
FEDOT supports a full cycle of machine learning task life that includes preprocessing, model selection, tuning, cross validation and serialization.

.. code-block:: python
model = Fedot(problem='classification')
model.fit(features=x_train, target=y_train)
prediction = model.predict(features=x_test)
metrics = model.get_metrics()
.. code-block:: python
Once FEDOT finds the best solution you have an opportunity to save it[link] and look closer to the solution and optimization process if needed.



Framework mostly works with sklearn, statsmodels and keras libraries
