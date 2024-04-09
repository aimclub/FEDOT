Operation
=========

This module provides a base class for operations in nodes.

:class:`Operation`:
-------------------

Base class for operations in nodes. Operations could be machine learning (or statistical) models or data operations.

Parameters:
    - operation_type (str): Name of the operation.

Methods:
    - __init__(self, operation_type): Initializes the Operation class.
    - fit(self, params, data): Defines and runs the evaluation strategy to train the operation with the data provided.
    - predict(self, fitted_operation, data, params, output_mode): Defines and runs the evaluation strategy to predict with the data provided.
    - predict_for_fit(self, fitted_operation, data, params, output_mode): Defines and runs the evaluation strategy to predict with the data provided during the fit stage.
    - assign_tabular_column_types(output_data, output_mode): Assigns types for columns based on task and output mode.
    - __str__(self): Returns string representation of the operation.
    - to_json(self): Serializes object and ignores unrelevant fields.

