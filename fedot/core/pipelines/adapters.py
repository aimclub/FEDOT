from copy import deepcopy
from typing import Any, Optional, Dict

from fedot.core.adapter import BaseOptimizationAdapter
from fedot.core.dag.graph_utils import map_dag_nodes
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.node import PipelineNode, Node
from fedot.core.pipelines.pipeline import Pipeline


class PipelineAdapter(BaseOptimizationAdapter[Pipeline]):
    """Optimization adapter for Pipeline<->OptGraph translation.
    It does 2 things:
    - on restore: recreate Pipeline Nodes from information stored in OptNodes
    - on adapt: create light-weight OptGraph (without 'heavy' data like
        fitted models) that can be used for reconstructing Pipelines.
    """

    def __init__(self, use_input_preprocessing: bool = True):
        super().__init__(base_graph_class=Pipeline)

        self.use_input_preprocessing = use_input_preprocessing

    @staticmethod
    def _transform_to_opt_node(node: Node) -> OptNode:
        # Prepare content for nodes, leave only simple data
        operation_name = str(node.operation)
        content = {'name': operation_name,
                   'params': node.parameters,
                   'metadata': node.metadata}
        return OptNode(deepcopy(content))

    @staticmethod
    def _transform_to_pipeline_node(node: OptNode) -> Node:
        # deepcopy to avoid accidental information sharing between opt graphs & pipelines
        content = deepcopy(node.content)
        return PipelineNode(operation_type=content['name'], content=content)

    def _adapt(self, adaptee: Pipeline) -> OptGraph:
        adapted_nodes = map_dag_nodes(self._transform_to_opt_node, adaptee.nodes)
        return OptGraph(adapted_nodes)

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> Pipeline:
        restored_nodes = map_dag_nodes(self._transform_to_pipeline_node, opt_graph.nodes)
        pipeline = Pipeline(restored_nodes, use_input_preprocessing=self.use_input_preprocessing)

        metadata = metadata or {}
        pipeline.computation_time = metadata.get('computation_time_in_seconds')

        return pipeline
