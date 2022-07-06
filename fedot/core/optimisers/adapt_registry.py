from inspect import signature, Parameter
from typing import Optional, Callable, Any

from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.utilities.singleton import SingletonMeta


# TODO: tests for
#  - graph argument in different places .
#  - no graph argument
#  - several graph arguments
#  - already adapted graph arg
# TODO: tests for
#  - decorator before init; with normal call
#  - decorator before init; with too-early call


class AdaptRegistry(metaclass=SingletonMeta):

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None):
        self.adapter = adapter or DirectAdapter()
        self._adaptee_cls = self.adapter._base_graph_class
        self._graph_cls = OptGraph

    def adapt(self, fun: Callable):
        return _transform(fun, self._maybe_adapt, self._maybe_restore)

    def restore(self, fun: Callable):
        return _transform(fun, self._maybe_restore, self._maybe_adapt)

    def _maybe_adapt(self, item):
        return self.adapter.adapt(item) if isinstance(item, self._adaptee_cls) else item

    def _maybe_restore(self, item):
        return self.adapter.restore(item) if isinstance(item, self._graph_cls) else item


def adapt(fun: Callable) -> Callable:
    return AdaptRegistry().adapt(fun)


def restore(fun: Callable) -> Callable:
    return AdaptRegistry().restore(fun)


def _transform(fun, f_args, f_ret):
    if not isinstance(fun, Callable):
        raise ValueError(f'Expected Callable, got {type(fun)}')

    def adapted_fun(*args, **kwargs):
        adapted_args = (f_args(arg) for arg in args)
        adapted_kwargs = dict((kw, f_args(arg)) for kw, arg in kwargs)
        result = fun(*adapted_args, **adapted_kwargs)
        return f_ret(result)

    return adapted_fun
