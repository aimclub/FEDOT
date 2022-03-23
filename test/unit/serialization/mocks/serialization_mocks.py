from fedot.core.operations.data_operation import DataOperation


class MockOperation:
    def __init__(self, operation_type='op', **kwargs):
        self.operation_type = operation_type
        self.operations_repo = 'operations_repo'

    def __eq__(self, other):
        return self.operation_type == other.operation_type


class MockNode:
    def __init__(self, name: str, nodes_from: list = None):
        self.name = name
        self.uid = name
        self._nodes_from = nodes_from or []
        self.content = {
            'name': DataOperation('test_operation')
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
        self.nodes = nodes if nodes else []
        self.operator = 'operator'

    def __eq__(self, other):
        return (
            self.operator == other.operator and
            self.nodes == other.nodes
        )
