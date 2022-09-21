from abc import abstractmethod
from copy import deepcopy
from typing import TypeVar, Generic, Type, Optional, Dict, Any, Callable, Tuple, Sequence, Union

from fedot.core.log import default_log
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.adapter.adapt_registry import AdaptRegistry

DomainStructureType = TypeVar('DomainStructureType')


class BaseOptimizationAdapter(Generic[DomainStructureType]):
    def __init__(self, base_graph_class: Type[DomainStructureType]):
        self._log = default_log(self)
        self.domain_graph_class = base_graph_class
        self.opt_graph_class = OptGraph

    def restore_func(self, fun: Callable) -> Callable:
        """Wraps native function so that it could accept domain graphs as arguments.

        Behavior: `restore( f(OptGraph) ) => f'(DomainGraph)`

        :param fun: native function that accepts native args (i.e. optimization graph)
         and requires adaptation of domain graph.

        :return: domain function that can be used inside Optimizer
        """
        return _transform(fun, f_args=self.adapt, f_ret=self.restore)

    def adapt_func(self, fun: Callable) -> Callable:
        """Wraps domain function so that it could accept native optimization graphs
        as arguments. If the function was registered as native, it is returned as-is.

        Behavior: `adapt( f(DomainGraph)->DomainGraph ) => f'(OptGraph)->OptGraph`

        :param fun: domain function that accepts domain args and required call to restore

        :return: native function that can be used inside Optimizer
        """
        if AdaptRegistry.is_native(fun):
            return fun
        return _transform(fun, f_args=self.restore, f_ret=self.adapt)

    def adapt(self, item: Union[DomainStructureType, Sequence[DomainStructureType]]) \
            -> Union[OptGraph, Sequence[OptGraph]]:
        if type(item) is self.domain_graph_class:
            return self._adapt(item)
        elif isinstance(item, Sequence) and type(item[0]) is self.domain_graph_class:
            return [self._adapt(graph) for graph in item]
        else:
            return item

    def restore(self, item: Union[OptGraph, Individual, PopulationT]) \
            -> Union[DomainStructureType, Sequence[DomainStructureType]]:
        if type(item) is self.opt_graph_class:
            return self._restore(item)
        elif isinstance(item, Individual):
            return self._restore(item.graph, item.metadata)
        elif isinstance(item, Sequence) and type(item[0]) is Individual:
            return [self._restore(ind.graph, ind.metadata) for ind in item]
        else:
            return item

    @abstractmethod
    def _adapt(self, adaptee: DomainStructureType) -> OptGraph:
        raise NotImplementedError()

    @abstractmethod
    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> DomainStructureType:
        raise NotImplementedError()


class DirectAdapter(BaseOptimizationAdapter[DomainStructureType]):
    """Naive optimization adapter for arbitrary class that just overwrites __class__."""

    def __init__(self,
                 base_graph_class: Type[DomainStructureType] = OptGraph,
                 base_node_class: Type = OptNode):
        super().__init__(base_graph_class)
        self._base_node_class = base_node_class

    def _adapt(self, adaptee: DomainStructureType) -> OptGraph:
        opt_graph = deepcopy(adaptee)
        opt_graph.__class__ = OptGraph

        for node in opt_graph.nodes:
            node.__class__ = OptNode
        return opt_graph

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> DomainStructureType:
        obj = deepcopy(opt_graph)
        obj.__class__ = self.domain_graph_class
        for node in obj.nodes:
            node.__class__ = self._base_node_class
        return obj


def _transform(fun: Callable, f_args: Callable, f_ret: Callable) -> Callable:
    """Transforms function in such a way that ``f_args`` is called on ``fun`` arguments
    and ``f_ret`` is called on the return value of original function.

    :param fun: function to be transformed
    :param f_args: arguments transformation function
    :param f_ret: return value transformation function
    :return: transformed function
    """

    if not isinstance(fun, Callable):
        raise ValueError(f'Expected Callable, got {type(fun)}')

    def adapted_fun(*args, **kwargs):
        adapted_args = (f_args(arg) for arg in args)
        adapted_kwargs = dict((kw, f_args(arg)) for kw, arg in kwargs.items())

        result = fun(*adapted_args, **adapted_kwargs)

        if result is None:
            adapted_result = None
        elif isinstance(result, Tuple):
            # In case when function returns not only Graph
            adapted_result = (f_ret(result_item) for result_item in result)
        else:
            adapted_result = f_ret(result)
        return adapted_result

    return adapted_fun
