from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.utils import DEFAULT_PARAMS_STUB, NESTED_PARAMS_LABEL


class _FittedOperationWithParams:
    def __init__(self, params):
        self._params = params

    def get_params(self):
        return self._params


def test_pipeline_node_parameters_setter_normalizes_default_and_nested_params():
    default_node = PipelineNode(operation_type='ridge')
    nested_node = PipelineNode(operation_type='ridge')

    default_node.parameters = DEFAULT_PARAMS_STUB
    nested_node.parameters = {NESTED_PARAMS_LABEL: {'alpha': 1.0}}

    assert default_node.parameters == {}
    assert nested_node.parameters['alpha'] == 1.0


def test_pipeline_node_update_params_uses_typed_merge_rule():
    node = PipelineNode(operation_type='ridge')
    node.parameters = {'alpha': 1.0}
    fitted_params = OperationParameters(alpha=1.0)
    fitted_params.update(beta=2.0)
    node.fitted_operation = _FittedOperationWithParams(fitted_params)

    node.update_params()

    assert node.parameters['alpha'] == 1.0
    assert node.parameters['beta'] == 2.0
