Atomized Model Template
========================

This module provides a class for creating templates of atomized models.

:class:`AtomizedModelTemplate`:
--------------------------------

Class for creating templates of atomized models.

Parameters:
    - node: Current node.
    - operation_id: Operation ID.
    - nodes_from: IDs of parent operations.
    - path: Path where to save parent JSON operation.

Methods:
    - __init__(self, node, operation_id, nodes_from, path): Initializes the AtomizedModelTemplate class.
    - convert_to_dict(self): Converts object parameters to a dictionary.
    - export_operation(self, path): Exports the operation.
    - import_json(self, operation_object): Imports a JSON-like object.

