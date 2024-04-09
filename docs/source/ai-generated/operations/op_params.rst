Operation Parameters
=====================

This module provides a class for storing parameters for models and data operations implementations.

:class:`OperationParameters`:
-----------------------------

Stores parameters for models and data operations implementations and records what parameters were changed.

Parameters:
    - operation_type (str): Type of the operation defined in the operation repository.
    - parameters (dict): Dictionary with parameters.

Methods:
    - __init__(self): Initializes the OperationParameters class.
    - from_operation_type(operation_type): Initializes parameters from the operation type.
    - update(self): Updates parameters.
    - get(self, key, default_value): Gets a parameter value.
    - setdefault(self, key, value): Sets a default value for a parameter.
    - to_dict(self): Converts parameters to a dictionary.
    - keys(self): Returns keys of changed parameters.
    - changed_parameters(self): Returns changed parameters.

