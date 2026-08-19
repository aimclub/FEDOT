import pytest
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.graph import OptNode

from fedot.core.pipelines.pipeline_advisor import PipelineChangeAdvisor
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
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
    assert new_primary_node.content['name'] in node_factory.graph_model_repository.get_operations(is_primary=True)
    assert new_intermediate_node.content['name'] in node_factory.graph_model_repository.get_operations(
        is_primary=False) and new_intermediate_node.content['name'] != intermediate_node.content['name']
    assert new_secondary_node.content['name'] in node_factory.graph_model_repository.get_operations(is_primary=False)


def test_get_intermediate_parent_node(nodes, node_factory):
    _, _, secondary_node = nodes
    new_intermediate_parent_node = node_factory.get_parent_node(secondary_node, is_primary=False)

    assert new_intermediate_parent_node is not None
    assert new_intermediate_parent_node.content['name'] in node_factory.graph_model_repository.get_operations(
        is_primary=False)
    assert new_intermediate_parent_node.content['name'] != secondary_node.content['name']
    assert new_intermediate_parent_node.content['name'] not in [
        str(n.content['name']) for n in secondary_node.nodes_from]


def test_get_separate_parent_node(nodes, node_factory):
    _, _, secondary_node = nodes
    new_separate_parent_node = node_factory.get_parent_node(secondary_node, is_primary=True)

    assert new_separate_parent_node is not None
    assert new_separate_parent_node.content['name'] in node_factory.graph_model_repository.get_operations(
        is_primary=True)
    assert new_separate_parent_node.content['name'] != secondary_node.content['name']


def test_get_child_node(node_factory):
    new_child_node = node_factory.get_node(is_primary=False)

    assert new_child_node is not None
    assert new_child_node.content['name'] in node_factory.graph_model_repository.get_operations(is_primary=False)


def test_get_primary_node(node_factory):
    new_primary_node = node_factory.get_node(is_primary=True)

    assert new_primary_node is not None
    assert new_primary_node.content['name'] in node_factory.graph_model_repository.get_operations(is_primary=True)


def test_get_final_node_proposes_models_only(node_factory):
    """The sink of a pipeline must be a model; the factory must never propose
    a data operation there, because verification would reject the offspring."""
    for _ in range(50):
        node = node_factory.get_final_node()
        assert node is not None
        assert node.content['name'] in ('dt', 'logit', 'rf')


def test_advisor_accepts_only_models_as_sink():
    advisor = PipelineChangeAdvisor(Task(TaskTypesEnum.classification))
    assert advisor.can_be_sink(OptNode(content={'name': 'rf'}))
    assert advisor.can_be_sink(OptNode(content={'name': 'logit'}))
    assert not advisor.can_be_sink(OptNode(content={'name': 'scaling'}))
    assert not advisor.can_be_sink(OptNode(content={'name': 'pca'}))

def test_change_node_proposes_operations_excluded_by_default_tags():
    """An explicit list of operations must reach node replacement even for
    operations the default repository tag filter hides (qda, mlp, dt)."""
    task = Task(TaskTypesEnum.classification)
    operations = ['logit', 'rf', 'qda', 'mlp', 'dt', 'scaling']
    requirements = PipelineComposerRequirements(primary=operations,
                                                secondary=operations)
    factory = PipelineOptNodeFactory(requirements=requirements,
                                     advisor=PipelineChangeAdvisor(task))
    node = OptNode(content={'name': 'logit'})

    proposed = {factory.exchange_node(node).content['name'] for _ in range(200)}

    assert {'qda', 'mlp', 'dt'}.issubset(proposed)


def test_from_available_operations_keeps_the_list_verbatim():
    """The repository must not re-filter an explicit operation list by preset:
    the caller resolves the preset when the user gave no list of their own."""
    task = Task(TaskTypesEnum.classification)
    operations = ['logit', 'rf', 'qda', 'mlp', 'dt', 'scaling']
    repository = PipelineOperationRepository()

    repository.from_available_operations(task=task, preset='best_quality',
                                         available_operations=operations)

    assert set(repository.get_operations(is_primary=True)) == set(operations)
    assert set(repository.get_operations(is_primary=False)) == set(operations)

