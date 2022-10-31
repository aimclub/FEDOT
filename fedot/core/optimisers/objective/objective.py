import itertools
from numbers import Real
from typing import Any, Optional, Union, Iterable, Callable, Sequence, TypeVar, Dict, Tuple

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.fitness import *
from fedot.core.repository.quality_metrics_repository import \
    MetricType, MetricsRepository, ComplexityMetricsEnum, MetricsEnum
from fedot.core.utilities.data_structures import ensure_wrapped_in_sequence

G = TypeVar('G', bound=Graph, covariant=True)
R = TypeVar('R', contravariant=True)
GraphFunction = Callable[[G], R]
ObjectiveFunction = GraphFunction[G, Fitness]


class Objective:
    """Represents objective function for computing metric values
    on Graphs and keeps information about metrics used."""

    def __init__(self,
                 quality_metrics: Dict[Any, Callable],
                 complexity_metrics: Optional[Dict[Any, Callable]] = None,
                 is_multi_objective: bool = False,
                 ):
        self._log = default_log(self)
        self.is_multi_objective = is_multi_objective
        self.quality_metrics = quality_metrics
        self.complexity_metrics = complexity_metrics or {}

    def __call__(self, graph: Graph, **metrics_kwargs: Any) -> Fitness:
        evaluated_metrics = []
        for metric_id, metric_func in self.metrics:
            try:
                metric_value = metric_func(graph, **metrics_kwargs)
                evaluated_metrics.append(metric_value)
            except Exception as ex:
                self._log.error(f'Objective evaluation error for graph {graph} on metric {metric_id}: {ex}')
                return null_fitness()  # fail right away
        return to_fitness(evaluated_metrics, self.is_multi_objective)

    @property
    def metrics(self) -> Iterable[Tuple[Any, Callable]]:
        return itertools.chain(self.quality_metrics.items(), self.complexity_metrics.items())

    @property
    def metric_names(self) -> Sequence[str]:
        return [str(metric_id) for metric_id, _ in self.metrics]


class MetricsObjective(Objective):
    def __init__(self,
                 metrics: Union[MetricType, Iterable[MetricType]],
                 is_multi_objective: bool = False):
        quality_metrics = {}
        complexity_metrics = {}

        for metric in ensure_wrapped_in_sequence(metrics):
            if isinstance(metric, MetricsEnum):
                metric_func = MetricsRepository().metric_by_id(metric)

                if isinstance(metric, ComplexityMetricsEnum):
                    complexity_metrics[metric] = metric_func
                else:
                    quality_metrics[metric] = metric_func
            elif isinstance(metric, Callable):
                metric_id = getattr(metric, '__name__', str(metric))
                quality_metrics[metric_id] = metric

        super().__init__(quality_metrics, complexity_metrics, is_multi_objective)


def to_fitness(metric_values: Optional[Sequence[Real]], multi_objective: bool = False) -> Fitness:
    if metric_values is None:
        return null_fitness()
    elif multi_objective:
        return MultiObjFitness(values=metric_values,
                               weights=[-1] * len(metric_values))
    else:
        return SingleObjFitness(*metric_values)
