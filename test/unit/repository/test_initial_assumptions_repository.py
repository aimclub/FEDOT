import json

import pytest

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.initial_assumptions_repository import (
    load_initial_assumption,
    load_initial_assumption_profile,
    load_initial_assumptions_repository,
)
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


def test_every_initial_assumption_loads_and_round_trips():
    repository = load_initial_assumptions_repository()
    operations_repository = OperationTypesRepository('all')

    for pipeline_id, description in repository['pipelines'].items():
        pipeline = load_initial_assumption(pipeline_id)
        task_type = TaskTypesEnum[description['task_type']]
        expected_nodes = {
            node['operation_type']: node for node in description['graph']['nodes']
        }
        actual_nodes = {node.operation.operation_type: node for node in pipeline.nodes}

        assert set(actual_nodes) == set(expected_nodes)
        for operation_type, expected_node in expected_nodes.items():
            operation_info = operations_repository.operation_info_by_id(operation_type)
            assert operation_info is not None
            assert task_type in operation_info.task_type
            actual_parameters = actual_nodes[operation_type].parameters
            assert expected_node['custom_params'].items() <= actual_parameters.items()

        serialized, _ = pipeline.save()
        restored = Pipeline().load(json.loads(serialized))
        assert {node.operation.operation_type for node in restored.nodes} == set(expected_nodes)


@pytest.mark.parametrize(
    'profile_id, expected_roots',
    [
        (
            'classification_narrow_many_class',
            ['xgboost', 'rf'],
        ),
        (
            'regression_medium_numeric',
            ['lgbmreg', 'catboostreg'],
        ),
    ],
)
def test_initial_assumption_profile_loads_expected_roots(profile_id, expected_roots):
    pipelines = load_initial_assumption_profile(profile_id)

    assert [pipeline.root_node.operation.operation_type for pipeline in pipelines] == expected_roots


def test_unknown_initial_assumption_ids_are_rejected():
    with pytest.raises(ValueError, match='Unknown initial assumption pipeline'):
        load_initial_assumption('missing')
    with pytest.raises(ValueError, match='Unknown initial assumption profile'):
        load_initial_assumption_profile('missing')
