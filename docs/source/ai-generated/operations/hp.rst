Hyperparameters Preprocessor
============================

This module provides a class for preprocessing hyperparameters before fitting operations.

:class:`HyperparametersPreprocessor`:
--------------------------------------

Class for hyperparameters preprocessing before operation fitting.

Parameters:
    - operation_type (str): Name of the operation.
    - n_samples_data (int): Number of rows in data.

Methods:
    - __init__(self, operation_type, n_samples_data): Initializes the HyperparametersPreprocessor class.
    - correct(self, params): Corrects hyperparameters based on preprocessing rules.

