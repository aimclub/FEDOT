class MockOperation:
    def __init__(self, operation_type='op', **kwargs):
        self.operation_type = operation_type
        self.operations_repo = 'operations_repo'

    def __eq__(self, other):
        return self.operation_type == other.operation_type


class MockPipelineTemplate:
    def __init__(self, struct_id='id'):
        self.struct_id = struct_id
        self.operation_templates = 'operation_templates'

    def __eq__(self, other):
        return self.struct_id == other.struct_id


class MockNode:
    def __init__(self, name: str, nodes_from: list = None):
        self.name = name
        self.nodes_from = nodes_from if nodes_from else []
        self._operator = '_operator'

    def __eq__(self, other):
        return (
            self.name == other.name and
            self.nodes_from == other.nodes_from and
            self._operator == other._operator
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
