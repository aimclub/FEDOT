from golem.core.optimisers.graph import OptNode
from golem.core.optimisers.opt_graph_builder import OptGraphBuilder, merge_opt_graph_builders

from fedot.core.pipelines.adapters import PipelineAdapter


class PipelineBuilder(OptGraphBuilder):
    def __init__(self, *initial_nodes: OptNode, **kwargs):
        super().__init__(PipelineAdapter(**kwargs), *initial_nodes)


merge_pipeline_builders = merge_opt_graph_builders
