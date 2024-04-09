Data Operation
==============

This module defines a class with fit/predict methods for evaluating data operations.

:class:`DataOperation`:
-----------------------

Class with `fit`/`predict` methods defining the evaluation strategy for the task.

Parameters:
    - operation_type (str): Name of the data operation.

Methods:
    - __init__(self, operation_type): Initializes the DataOperation class.
    - metadata(self): Provides metadata for the DataOperation.
    - assign_tabular_column_types(output_data, output_mode): Assigns column types for tabular data.

