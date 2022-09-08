from abc import ABC
from typing import TypeVar, Generic

from fedot.core.dag.graph import Graph
from fedot.core.optimisers.fitness import Fitness
from .objective import Objective

G = TypeVar('G', bound=Graph, covariant=True)


class ObjectiveEvaluate(ABC, Generic[G]):
    """Defines how Objective must be evaluated on Graphs.

     Responsibilities:
     - Graph-specific evaluation policy: typically, Graphs require some kind of evaluation
     before Objective could be estimated on them. E.g. Machine-learning pipelines must be
     fit on train data before they could be evaluated on the test data.
     - Objective-specific estimation: typically objectives require additional parameters
     besides Graphs for estimation, e.g. test data for estimation of prediction quality.
     - Optionally, compute additional statistics for Graphs (intermediate metrics).

     Default implementation is just a closure that calls :param objective: with
      redirected keyword arguments :param objective_kwargs:
    """

    def __init__(self, objective: Objective, eval_n_jobs: int = 1, **objective_kwargs):
        self._objective = objective
        self._objective_kwargs = objective_kwargs
        self._eval_n_jobs = eval_n_jobs

    def __call__(self, graph: G) -> Fitness:
        """Provides functional interface for ObjectiveEvaluate."""
        return self.evaluate(graph)

    def evaluate(self, graph: G) -> Fitness:
        """Evaluate graph and compute its fitness."""
        return self._objective(graph, **self._objective_kwargs)

    def evaluate_intermediate_metrics(self, graph: G):
        """Compute intermediate metrics for each graph node and store it there."""
        pass
