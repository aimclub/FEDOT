Model
=====

This module defines a class with fit/predict methods for evaluating the model.

:class:`Model`:
---------------

Class with `fit`/`predict` methods defining the evaluation strategy for the task.

Parameters:
    - operation_type (str): Name of the model.

Methods:
    - __init__(self, operation_type): Initializes the Model class.
    - assign_tabular_column_types(output_data, output_mode): Assigns types for tabular data obtained from model predictions.

