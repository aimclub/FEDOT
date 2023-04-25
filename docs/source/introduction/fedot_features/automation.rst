AutoML capabilities
===================

FEDOT is capable of setting its 'automation rate' by omitting some of its parameters.
For example, if you just create the FEDOT instance and call the ``fit`` method with the appropriate dataset on it,
you will have a full automation of the learning process,
see :doc:`automated composing </introduction/tutorial/composing_pipelines/automated_creation>`

At the same time, if you pass some of the parameters, you will have a partial automation,
see :doc:`manual composing </introduction/tutorial/composing_pipelines/manual_creation>`.

Other than that, you can utilize more AutoML means.


Data nature
-----------

FEDOT uses specific data processing according to the source
of the input data (whether it is pandas DataFrame, or numpy array, or just a path to the dataset, etc).

.. note::

    Be careful with datetime features, as they are to be casted into a float type with milliseconds unit.


Apart from that FEDOT is capable of working with multi-modal data.
It means that you can pass it different types of datasets
(tables, texts, images, etc) and it will get the information from within to work with it.

.. seealso::
    :doc:`Detailed multi-modal data description and usage </basics/multi_modal_tasks>`


Dimensional operations
----------------------

FEDOT supports bunch of dimensionality preprocessing operations that can be be added to the pipeline as a node.

Feature selection
^^^^^^^^^^^^^^^^^

There are different linear and non-linear algorithms for regression and classification tasks
which uses scikit-learn's Recursive Feature Elimination (RFE).

.. list-table:: Feature selection operations
   :header-rows: 1

   * - API name
     - Definition
   * - rfe_lin_reg
     - RFE via Linear Regression regressor
   * - rfe_non_lin_reg
     - RFE via Decision tree regressor
   * - rfe_lin_class
     - RFE via Logistic Regression classifier
   * - rfe_non_lin_class
     - RFE via Decision tree classifier

Feature extraction
^^^^^^^^^^^^^^^^^^

These algorithms are used for generating new features.

.. list-table:: Feature extraction operations
   :header-rows: 1

   * - API name
     - Definition
   * - pca
     - Principal Component Analysis (PCA)
   * - kernel_pca
     - Principal Component Analysis (PCA) with kernel methods
   * - fast_ica
     - Fast Independent Component Analysis (FastICA)
   * - poly_features
     - Polynomial Features transformations
   * - lagged
     - Time-series to table transformation
   * - sparse_lagged
     - Time-series to sparse table transformation

Feature expansion
^^^^^^^^^^^^^^^^^

These methods expands specific features to a bigger amount

.. list-table:: Feature expansion operations
   :header-rows: 1

   * - API name
     - Definition
   * - one_hot_encoding
     - One-hot encoding


Models used
-----------

Using the parameter ``preset`` of the :doc:`main API </api/api>` you can specify
what models are available during the learning process. 

It influences:

* composing speed and quality
* computational behaviour
* task relevance

For example, ``'best_quality'`` option allows FEDOT to use entire list of available models for a specified task.
In contrast ``'fast_train'`` ensures only fast learning models are going to be used.

Apart from that there are other options whose names speak for themselves: ``'stable'``, ``'auto'``, ``'gpu'``, ``'ts'``,
``'automl'`` (the latter uses only AutoML models as pipeline nodes).

.. note::
    To make it simple, FEDOT uses ``auto`` by default to identify the best choice for you.