Intro to FEDOT
==============

.. |FEDOT logo| image:: img_intro/fedot_logo.png
   :width: 100%

.. |Pipeline schema| image:: img_intro/pipeline_small.png
   :width: 100%

.. |Example of solution| image:: img_intro/pipeline.png
   :width: 100%

FEDOT - an open-source framework for automated modeling and machine learning (AutoML). It produces a lightweight end-to-end solution in an automated way using an evolutionary approach.

|FEDOT logo|

FEDOT supports classification (binary and multiclass), regression, clustering, and time series forecasting tasks. FEDOT works both on unimodal (only tabular/image/text data) and multimodal data (more than one data source).

|Pipeline schema|

FEDOT supports a full cycle of machine learning task life that includes preprocessing, model selection, tuning, cross validation and serialization.

.. code-block:: python

    model = Fedot(problem='classification')
    model.fit(features=x_train, target=y_train)
    prediction = model.predict(features=x_test)
    metrics = model.get_metrics()

Once FEDOT finds the best solution you have an opportunity to save it[link] and look closer to the solution and optimization process if needed.

|Example of solution|

Framework mostly works with sklearn, statsmodels and keras libraries.

To see more information about FEDOT features go to the `features <https://fedot.readthedocs.io/en/latest/introduction/fedot_features.html>`_ page.
To see a quickstart guide go to the `quickstart <https://fedot.readthedocs.io/en/latest/introduction/tutorial/quickstart.html>`_ page.
