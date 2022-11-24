import pytest
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.graph import OptNode

from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture(scope='module')
def nodes():
    primary_node = OptNode(content={'name': 'pca'})
    intermediate_node = OptNode(content={'name': 'dt'},
                                nodes_from=[primary_node])
    secondary_node = OptNode(content={'name': 'logit'},
                             nodes_from=[intermediate_node])
    return primary_node, intermediate_node, secondary_node


@pytest.fixture(scope='module')
def node_factory():
    task = Task(TaskTypesEnum.classification)
    advisor = PipelineChangeAdvisor(task)
    primary_operations = ['bernb', 'rf', 'qda', 'pca', 'normalization']
    secondary_operations = ['dt', 'logit', 'rf', 'scaling']
    requirements = PipelineComposerRequirements(primary=primary_operations,
                                                secondary=secondary_operations)
    return PipelineOptNodeFactory(requirements=requirements,
                                  advisor=advisor)


def test_change_node(nodes, node_factory):
    primary_node, intermediate_node, secondary_node = nodes
    new_primary_node = node_factory.exchange_node(primary_node)
    new_intermediate_node = node_factory.exchange_node(intermediate_node)
    new_secondary_node = node_factory.exchange_node(secondary_node)

    assert new_primary_node is not None
    assert new_secondary_node is not None
    assert new_intermediate_node is not None
    assert new_primary_node.content['name'] \
           in node_factory.graph_model_repository.get_operations(is_primary=True)
    assert new_intermediate_node.content['name'] \
           in node_factory.graph_model_repository.get_operations(is_primary=False) and \
           new_intermediate_node.content['name'] != intermediate_node.content['name']
    assert new_secondary_node.content['name'] \
           in node_factory.graph_model_repository.get_operations(is_primary=False)


def test_get_intermediate_parent_node(nodes, node_factory):
    _, _, secondary_node = nodes
    new_intermediate_parent_node = node_factory.get_parent_node(secondary_node, is_primary=False)

    assert new_intermediate_parent_node is not None
    assert new_intermediate_parent_node.content['name'] \
           in node_factory.graph_model_repository.get_operations(is_primary=False)
    assert new_intermediate_parent_node.content['name'] != secondary_node.content['name']
    assert new_intermediate_parent_node.content['name'] \
           not in [str(n.content['name']) for n in secondary_node.nodes_from]


def test_get_separate_parent_node(nodes, node_factory):
    _, _, secondary_node = nodes
    new_separate_parent_node = node_factory.get_parent_node(secondary_node, is_primary=True)

    assert new_separate_parent_node is not None
    assert new_separate_parent_node.content['name'] \
           in node_factory.graph_model_repository.get_operations(is_primary=True)
    assert new_separate_parent_node.content['name'] != secondary_node.content['name']


def test_get_child_node(node_factory):
    new_child_node = node_factory.get_node(is_primary=False)

    assert new_child_node is not None
    assert new_child_node.content['name'] \
           in node_factory.graph_model_repository.get_operations(is_primary=False)


def test_get_primary_node(node_factory):
    new_primary_node = node_factory.get_node(is_primary=True)

    assert new_primary_node is not None
    assert new_primary_node.content['name'] \
           in node_factory.graph_model_repository.get_operations(is_primary=True)
