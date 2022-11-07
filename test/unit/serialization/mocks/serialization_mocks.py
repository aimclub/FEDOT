from fedot.core.operations.data_operation import DataOperation


class MockOperation:
    def __init__(self, operation_type='op', **kwargs):
        self.operation_type = operation_type
        self.operations_repo = 'operations_repo'

    def __eq__(self, other):
        return self.operation_type == other.operation_type


def operation_to_json(self):
    """
    Uses regular serialization but excludes "operations_repo" field cause it has no any important info about class
    """
    return {
        k: v
        for k, v in sorted(vars(self).items())
        if k not in ['operations_repo']
    }


class MockNode:
    def __init__(self, name: str, nodes_from: list = None):
        self.name = name
        self.uid = name
        self._nodes_from = nodes_from or []
        self.content = {
            'name': 'test_operation'
        }

    @property
    def nodes_from(self):
        return self._nodes_from

    def __eq__(self, other):
        return (
            self.name == other.name and
            self._nodes_from == other._nodes_from
        )


class MockGraph:
    def __init__(self, nodes: list = None):
        self._nodes = nodes if nodes else []

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, new_nodes):
        self._nodes = new_nodes

    def __eq__(self, other):
        return self.nodes == other.nodes
