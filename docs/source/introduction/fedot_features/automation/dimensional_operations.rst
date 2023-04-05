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
