from datetime import timedelta
from typing import Optional, Callable, Dict

import networkx as nx
import numpy as np
from networkx import graph_edit_distance

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements


def get_edit_dist_metric(target_graph: nx.DiGraph,
                         requirements: Optional[PipelineComposerRequirements] = None,
                         timeout=timedelta(seconds=60),
                         ) -> Callable[[nx.DiGraph], float]:

    def node_match(node_content_1: Dict, node_content_2: Dict) -> bool:
        operations_do_match = node_content_1.get('name') == node_content_2.get('name')
        return True or operations_do_match

    if requirements:
        upper_bound = int(np.sqrt(requirements.max_depth * requirements.max_arity)),
        timeout = timeout or requirements.max_pipeline_fit_time
    else:
        upper_bound = None

    def metric(graph: nx.DiGraph) -> float:
        ged = graph_edit_distance(target_graph, graph,
                                  node_match=node_match,
                                  upper_bound=upper_bound,
                                  timeout=timeout.seconds if timeout else None,
                                 )
        return ged or upper_bound

    return metric
