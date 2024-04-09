Operation Factory
==================

This module provides a base class for determining what type of operations should be defined in the node.

:class:`OperationFactory`:
--------------------------

Base class for determining what type of operations should be defined in the node.

Parameters:
    - operation_name (str): Name of the operation.

Methods:
    - __init__(self, operation_name): Initializes the OperationFactory class.
    - get_operation(self): Returns the desired object of the 'Data_operation' or 'Model' class.
    - operation_type_name(self): Determines the type of operations for this node.

