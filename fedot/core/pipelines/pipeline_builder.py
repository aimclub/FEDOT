from functools import partial

from fedot.core.optimisers.opt_graph_builder import OptGraphBuilder, merge_opt_graph_builders
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import Node


PipelineBuilder = partial(OptGraphBuilder, PipelineAdapter())
merge_pipeline_builders = merge_opt_graph_builders
