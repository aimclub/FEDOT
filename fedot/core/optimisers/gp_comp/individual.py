import gc
from typing import List

from fedot.core.optimisers.graph import OptGraph
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.core.pipelines.pipeline import Pipeline

ERROR_PREFIX = 'Invalid graph configuration:'


class Individual:
    def __init__(self, graph: 'OptGraph', fitness: List[float] = None,
                 parent_operators: List[ParentOperator] = None):
        self.parent_operators = parent_operators if parent_operators is not None else []
        self.fitness = fitness

        # TODO remove direct pipeline reference
        self.graph = _release_pipeline_resources(graph) if isinstance(graph, Pipeline) else graph


def _release_pipeline_resources(pipeline: Pipeline):
    """
    Remove all 'heavy' parts from the graph
    :param pipeline: fitted pipeline
    :return: pipeline without fitted models and data
    """
    pipeline.unfit()
    gc.collect()
    return pipeline
