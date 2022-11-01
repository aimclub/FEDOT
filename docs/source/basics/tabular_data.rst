Tabular Data Prediction
==============================================

Introduction
~~~~~~~~~~~~

As common AutoML frameworks, FEDOT solves problems with data that are represented as tables.
FEDOT allows you to automate machine learning pipeline design for tabular data in ``classification`` and ``regression``
problems.

Also, it provides a high-level API that enables you to use common fit/predict interface. To use API it is required
to import certain object:

.. code-block:: python

    from fedot.api.main import Fedot

Loading training and test data from a CSV file as a Pandas dataframe ``pd.DataFrame``.

.. code-block:: python

    train = pd.DataFrame('train.csv')
    test = pd.DataFrame('test.csv')

Initialize the Fedot object and define the type of modeling problem. In this case, problem is ``classification``.

.. code-block:: python

    model = Fedot(problem='classification', metric='roc_auc')

.. note::

    Class ``Fedot()`` has more than two params, e.g. ``timeout`` for setting time limits or
    ``n_jobs`` for parallelization. For more details, see the :doc:`FEDOT API <api>` section in our documentation.

The ``fit()`` method begins the optimization and returns the resulting composite pipeline.

.. code-block:: python

    model.fit(features=train, target='target')

The ``predict()`` method, which uses an already fitted pipeline, returns values for the target.

.. code-block:: python

    prediction = model.predict(features=test)

.. hint::

    If you want to predict target probability use ``predict_proba()`` method.

The ``get_metrics()`` method estimates the quality of predictions according the selected metrics.

.. code-block:: python

    prediction = model.get_metrics()

.. note::

    The same way FEDOT can be used to ``regression`` problem. It is only required to change params according the problem
    in main class object:

    .. code-block:: python

        model = Fedot(problem='regression', metric='rmse')

Examples
~~~~~~~~

More details you can find in the follow links:

**Simple**

* `Classification using API <https://github.com/nccr-itmo/FEDOT/blob/master/examples/simple/classification/api_classification.py>`_
* `Regression using API <https://github.com/nccr-itmo/FEDOT/blob/master/examples/simple/regression/api_regression.py>`_
* `Classification with tuning <https://github.com/nccr-itmo/FEDOT/blob/master/examples/simple/classification/classification_with_tuning.py>`_
* `Regression with tuning <https://github.com/nccr-itmo/FEDOT/blob/master/examples/simple/regression/regression_with_tuning.py>`_

**Advanced**

* `Multiclass classification problem <https://github.com/nccr-itmo/FEDOT/blob/master/examples/simple/classification/multiclass_prediction.py>`_
* `Classification with unbalanced data <https://github.com/nccr-itmo/FEDOT/blob/master/examples/simple/classification/resample_examples.py>`_
* `Image classification problem <https://github.com/nccr-itmo/FEDOT/blob/master/examples/simple/classification/image_classification_problem.py>`_

**Cases**

* `Case: Credit scoring problem <https://github.com/nccr-itmo/FEDOT/blob/master/cases/credit_scoring/credit_scoring_problem.py>`_