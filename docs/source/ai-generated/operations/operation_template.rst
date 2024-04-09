Operation Template
===================

This module provides base classes for creating operation templates.

:class:`OperationTemplateAbstract`:
-----------------------------------

Base class used for creating different types of operation templates.

Methods:
    - __init__(self): Initializes the OperationTemplateAbstract class.
    - import_json(self, operation_object): Parses JSON-like object and fills local fields.
    - convert_to_dict(self): Converts all object parameters to a dictionary.

:class:`OperationTemplate`:
---------------------------

Class for creating operation templates.

Parameters:
    - node: Current node.
    - operation_id: Operation ID.
    - nodes_from: IDs of parent operations.

Methods:
    - __init__(self, node, operation_id, nodes_from): Initializes the OperationTemplate class.
    - export_operation(self, return_path): Exports the operation.
    - import_json(self, operation_object): Imports a JSON-like object.

