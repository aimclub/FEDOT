Dimensionality operations
-------------------------

FEDOT supports bunch of dimension preprocessing operations that can be be added to the pipeline as a node.

Feature selection
^^^^^^^^^^^^^^^^^

There are different linear and non-linear algorithms for regression and classification tasks
which uses scikit-learn's Recursive Feature Elimination (RFE).

Feature extraction
^^^^^^^^^^^^^^^^^^

Currently there are PCA (kernel methods supported), fast ICA and
polynomial features algorithms for generating new features. 