from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptNode
from fedot.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import DEFAULT_PARAMS_STUB


def test_default_node_factory():
    task = Task(TaskTypesEnum.classification)
    advisor = PipelineChangeAdvisor(task)
    primary_operations = ['bernb', 'rf', 'qda', 'pca', 'normalization']
    secondary_operations = ['dt', 'logit', 'rf', 'scaling']
    requirements = PipelineComposerRequirements(primary=primary_operations,
                                                secondary=secondary_operations)
    node_factory = DefaultOptNodeFactory(requirements=requirements)

    primary_node = OptNode(content={'name': 'pca',
                                    'params': DEFAULT_PARAMS_STUB})
    secondary_node = OptNode(content={'name': 'dt',
                                      'params': DEFAULT_PARAMS_STUB},
                             nodes_from=[primary_node])

    changed_primary_node = node_factory.exchange_node(primary_node)
    changed_secondary_node = node_factory.exchange_node(secondary_node)
    new_primary_node = node_factory.get_node(primary=True)
    new_secondary_node = node_factory.get_node(primary=False)
    new_separate_parent_node = node_factory.get_parent_node(secondary_node, primary=True)
    new_intermediate_parent_node = node_factory.get_parent_node(secondary_node, primary=False)

    for primary_node in [changed_primary_node, new_primary_node, new_separate_parent_node]:
        assert primary_node is not None
        assert primary_node.content['name'] in requirements.primary
    for secondary_node in [changed_secondary_node, new_secondary_node, new_intermediate_parent_node]:
        assert secondary_node is not None
        assert secondary_node.content['name'] in requirements.secondary
