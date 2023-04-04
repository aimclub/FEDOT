What is FEDOT
=============

.. |FEDOT logo| image:: img_intro/fedot_logo.png
   :width: 100%

.. |Pipeline schema| image:: img_intro/pipeline_small.png
   :width: 100%

.. |Example of solution| image:: img_intro/pipeline.png
   :width: 100%

FEDOT is an open-source framework for automated modeling and machine learning (AutoML). It produces a lightweight end-to-end ML solution in an automated way using an evolutionary approach.

|FEDOT logo|

FEDOT supports classification (binary and multiclass), regression, and time series forecasting tasks. FEDOT works both on unimodal (only tabular/image/text data) and multimodal data (more than one data source).

|Pipeline schema|

FEDOT supports a full life-сyсle of machine learning task that includes preprocessing, model selection, tuning, cross validation and serialization.

.. code-block:: python

    model = Fedot(problem='classification', timeout=5, preset='best_quality', n_jobs=-1)
    model.fit(features=x_train, target=y_train)
    prediction = model.predict(features=x_test)
    metrics = model.get_metrics(target=y_test)

Once FEDOT finds the best solution you have an opportunity :doc:`to save it </basics/pipeline_save_load>` and look closer to the solution and optimization process if needed.

|Example of solution|

Framework uses ML models mostly from sklearn, statsmodels and keras libraries.

You can investigate more FEDOT's concepts :doc:`here </basics/concepts>`.

There is a :doc:`quickstart guide for beginers </introduction/tutorial/quickstart>`.
