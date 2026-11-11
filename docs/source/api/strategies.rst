Strategies
==========

Operations in pipeline nodes select an evaluation strategy from the operation catalogue.
Strategies adapt calls to an estimator family; algorithm implementations and adapters
live separately in ``fedot.core.operations.implementations``. Model and data-operation
strategies are grouped in separate packages. Classification and regression variants
of the same family share a module.

Base contract
-------------

.. automodule:: fedot.core.operations.strategies.base
   :members:
   :no-undoc-members:

Model strategies
----------------

Sklearn estimators
^^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.models.sklearn
   :members:
   :no-undoc-members:

Container-based estimator adapters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.models.adapters
   :members:
   :no-undoc-members:

Tabular neural models
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.models.neural
   :members:
   :no-undoc-members:

Boosting
^^^^^^^^

.. automodule:: fedot.core.operations.strategies.models.boosting
   :members:
   :no-undoc-members:

Foundation models
^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.models.foundation
   :members:
   :no-undoc-members:

Forecasting
^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.models.forecasting
   :members:
   :no-undoc-members:

cuML
^^^^

.. automodule:: fedot.core.operations.strategies.models.cuml
   :members:
   :no-undoc-members:

Custom models
^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.models.custom
   :members:
   :no-undoc-members:

Data-operation strategies
-------------------------

Common preprocessing
^^^^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.data_operations.common_preprocessing
   :members:
   :no-undoc-members:

Classification and regression preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.data_operations.preprocessing
   :members:
   :no-undoc-members:

Projections
^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.data_operations.projection
   :members:
   :no-undoc-members:

Optional preprocessing
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.data_operations.optional
   :members:
   :no-undoc-members:

Time series
^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.data_operations.time_series
   :members:
   :no-undoc-members:

Text
^^^^

.. automodule:: fedot.core.operations.strategies.data_operations.text
   :members:
   :no-undoc-members:

Other integration strategies
----------------------------

Data sources
^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.sources
   :members:
   :no-undoc-members:

External operations
^^^^^^^^^^^^^^^^^^^

.. automodule:: fedot.core.operations.strategies.extensions
   :members:
   :no-undoc-members:

AutoML
^^^^^^

.. automodule:: fedot.core.operations.strategies.automl
   :members:
   :no-undoc-members:

The historical ``evaluation`` package has been removed. Integrations should use
the canonical paths listed above. Saved pipelines from earlier versions should
be run with the FEDOT version and environment used to train them; loading those
artifacts in the restructured version is not supported.
