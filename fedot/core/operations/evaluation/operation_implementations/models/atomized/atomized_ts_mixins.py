from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.random_pipeline_factory import RandomPipelineFactory
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository


class AtomizedTimeSeriesBuildFactoriesMixin:
    """ Add `build_factories` method that delete primary nodes from node factory """
    @classmethod
    def build_factories(cls, requirements, graph_generation_params):
        graph_model_repository = PipelineOperationRepository(operations_by_keys={'primary': requirements.secondary,
                                                                                 'secondary': requirements.secondary})
        node_factory = PipelineOptNodeFactory(requirements, graph_generation_params.advisor, graph_model_repository)
        random_pipeline_factory = RandomPipelineFactory(graph_generation_params.verifier, node_factory)
        return node_factory, random_pipeline_factory