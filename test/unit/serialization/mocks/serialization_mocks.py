from fedot.core.serializers import GraphNodeSerializer, GraphSerializer, OperationSerializer, PipelineTemplateSerializer


class MockOperation(OperationSerializer):
    def __init__(self, **kwargs):
        self.operations_repo = 'operations_repo'

    def __eq__(self, other):
        return self.operations_repo == other.operations_repo


class MockPipelineTemplate(PipelineTemplateSerializer):
    def __init__(self):
        self.operation_templates = 'operation_templates'

    def __eq__(self, other):
        return self.operation_templates == other.operation_templates


class MockNode(GraphNodeSerializer):
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


class MockGraph(GraphSerializer):
    def __init__(self, nodes: list = None):
        self.nodes = nodes if nodes else []
        self.operator = 'operator'

    def __eq__(self, other):
        return (
            self.operator == other.operator and
            self.nodes == other.nodes
        )
