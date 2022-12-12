from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, TypeVar, Generic, Type, Optional, Dict, Any, Callable, Tuple, Sequence, Union

from fedot.core.dag.graph import Graph
from fedot.core.log import default_log
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.adapter.adapt_registry import AdaptRegistry
from fedot.core.optimisers.opt_history_objects.individual import Individual

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.operators.operator import PopulationT

DomainStructureType = TypeVar('DomainStructureType')


class BaseOptimizationAdapter(Generic[DomainStructureType]):
    def __init__(self, base_graph_class: Type[DomainStructureType] = Graph):
        self._log = default_log(self)
        self.domain_graph_class = base_graph_class
        self.opt_graph_class = OptGraph

    def restore_func(self, fun: Callable) -> Callable:
        """Wraps native function so that it could accept domain graphs as arguments.

        Behavior: ``restore( f(OptGraph)->OptGraph ) => f'(DomainGraph)->DomainGraph``

        Implementation details.
        The method wraps callable into a function that transforms its args & return value.
        Arguments are transformed by ``adapt`` (that maps domain graphs to internal graphs).
        Return value is transformed by ``restore`` (that maps internal graphs to domain graphs).

        Args:
            fun: native function that accepts native args (i.e. optimization graph)

        Returns:
            Callable: domain function that can accept domain graphs
        """
        return _transform(fun, f_args=self.adapt, f_ret=self.restore)

    def adapt_func(self, fun: Callable) -> Callable:
        """Wraps domain function so that it could accept native optimization graphs
        as arguments. If the function was registered as native, it is returned as-is.
        ``AdaptRegistry`` is responsible for function registration.

        Behavior: ``adapt( f(DomainGraph)->DomainGraph ) => f'(OptGraph)->OptGraph``

        Implementation details.
        The method wraps callable into a function that transforms its args & return value.
        Arguments are transformed by ``restore`` (that maps internal graphs to domain graphs).
        Return value is transformed by ``adapt`` (that maps domain graphs to internal graphs).

        Args:
            fun: domain function that accepts domain graphs

        Returns:
            Callable: native function that can accept opt graphs
            and be used inside Optimizer
        """
        if AdaptRegistry.is_native(fun):
            return fun
        return _transform(fun, f_args=self.restore, f_ret=self.adapt)

    def adapt(self, item: Union[DomainStructureType, Sequence[DomainStructureType]]) \
            -> Union[OptGraph, Sequence[OptGraph]]:
        """Maps domain graphs to internal graph representation used by optimizer.
        Performs mapping only if argument has a type of domain graph.

        Args:
            item: a domain graph or sequence of them

        Returns:
            OptGraph | Sequence: mapped internal graph or sequence of them
        """
        if type(item) is self.domain_graph_class:
            return self._adapt(item)
        elif isinstance(item, Sequence) and type(item[0]) is self.domain_graph_class:
            return [self._adapt(graph) for graph in item]
        else:
            return item

    def restore(self, item: Union[OptGraph, Individual, PopulationT]) \
            -> Union[DomainStructureType, Sequence[DomainStructureType]]:
        """Maps graphs from internal representation to domain graphs.
        Performs mapping only if argument has a type of internal representation.

        Args:
            item: an internal graph representation or sequence of them

        Returns:
            OptGraph | Sequence: mapped domain graph or sequence of them
        """
        if type(item) is self.opt_graph_class:
            return self._restore(item)
        elif isinstance(item, Individual):
            return self._restore(item.graph, item.metadata)
        elif isinstance(item, Sequence) and isinstance(item[0], Individual):
            return [self._restore(ind.graph, ind.metadata) for ind in item]
        else:
            return item

    @abstractmethod
    def _adapt(self, adaptee: DomainStructureType) -> OptGraph:
        """Implementation of ``adapt`` for single graph."""
        raise NotImplementedError()

    @abstractmethod
    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> DomainStructureType:
        """Implementation of ``restore`` for single graph."""
        raise NotImplementedError()


class IdentityAdapter(BaseOptimizationAdapter[DomainStructureType]):
    """Identity adapter that performs no transformation, returning same graphs."""

    def _adapt(self, adaptee: DomainStructureType) -> OptGraph:
        return adaptee

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> DomainStructureType:
        return opt_graph


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
    """Wraps function by transforming its arguments and return value:
     ``f_args`` is called on each of the function arguments,
     ``f_ret`` is called on the return value of original function.

     This is a helper function used for adaption of callables by
     :py:class:`fedot.core.adapter.adapter.BaseOptimizationAdapter`.

    Args:
        fun: function to be transformed
        f_args: argument transformation function
        f_ret: return value transformation function

    Returns:
        Callable: wrapped transformed function
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
            adapted_result = tuple(f_ret(result_item) for result_item in result)
        else:
            adapted_result = f_ret(result)
        return adapted_result

    return adapted_fun
