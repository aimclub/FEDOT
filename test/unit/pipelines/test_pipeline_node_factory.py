import pytest

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptNode
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import DEFAULT_PARAMS_STUB


@pytest.fixture(scope='module')
def requirements():
    primary_operations = ['bernb', 'rf', 'qda', 'pca']
    secondary_operations = ['dt', 'logit', 'rf', 'scaling']
    requirements = PipelineComposerRequirements(primary=primary_operations,
                                                secondary=secondary_operations)
    return requirements


@pytest.fixture(scope='module')
def nodes():
    primary_node = OptNode(content={'name': 'pca',
                                    'params': DEFAULT_PARAMS_STUB})
    intermediate_node = OptNode(content={'name': 'dt',
                                         'params': DEFAULT_PARAMS_STUB},
                                nodes_from=[primary_node])
    secondary_node = OptNode(content={'name': 'logit',
                                      'params': DEFAULT_PARAMS_STUB},
                             nodes_from=[intermediate_node])
    return primary_node, intermediate_node, secondary_node


@pytest.fixture(scope='module')
def node_factory():
    return PipelineOptNodeFactory()


@pytest.fixture(scope='module')
def advisor():
    task = Task(TaskTypesEnum.classification)
    return PipelineChangeAdvisor(task)


def test_change_node(nodes, node_factory, requirements, advisor):
    primary_node, intermediate_node, secondary_node = nodes
    new_primary_node = node_factory.change_node(primary_node, requirements)
    new_intermediate_node = node_factory.change_node(intermediate_node, requirements, advisor)
    new_secondary_node = node_factory.change_node(secondary_node, requirements)

    assert new_primary_node is not None
    assert new_secondary_node is not None
    assert new_intermediate_node is not None
    assert new_primary_node.content['name'] in requirements.primary
    assert new_intermediate_node.content['name'] in requirements.secondary and \
           new_intermediate_node.content['name'] != intermediate_node.content['name']
    assert new_secondary_node.content['name'] in requirements.secondary


def test_get_intermediate_parent_node(nodes, node_factory, requirements, advisor):
    _, _, secondary_node = nodes
    new_intermediate_parent_node = node_factory.get_intermediate_parent_node(secondary_node, requirements, advisor)

    assert new_intermediate_parent_node is not None
    assert new_intermediate_parent_node.content['name'] in requirements.secondary
    assert new_intermediate_parent_node.content['name'] != secondary_node.content['name']
    assert new_intermediate_parent_node.content['name'] \
           not in [str(n.content['name']) for n in secondary_node.nodes_from]


def test_get_separate_parent_node(nodes, node_factory, requirements, advisor):
    _, _, secondary_node = nodes
    new_separate_parent_node = node_factory.get_separate_parent_node(secondary_node, requirements, advisor)

    assert new_separate_parent_node is not None
    assert new_separate_parent_node.content['name'] in requirements.primary
    assert new_separate_parent_node.content['name'] != secondary_node.content['name']


def test_get_child_node(node_factory, requirements):
    new_child_node = node_factory.get_child_node(requirements)

    assert new_child_node is not None
    assert new_child_node.content['name'] in requirements.secondary


def test_get_primary_node(node_factory, requirements):
    new_primary_node = node_factory.get_primary_node(requirements)

    assert new_primary_node is not None
    assert new_primary_node.content['name'] in requirements.primary
