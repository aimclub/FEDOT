from inspect import signature
from typing import Optional, Callable, Any

from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.utilities.singleton import SingletonMeta


class AdaptRegistry(metaclass=SingletonMeta):

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None):
        self._adapter = adapter or DirectAdapter()
        self._require_adapt = {}
        self._require_restore = {}

    def adapt(self, fun: Callable[..., Any], *args, **kwargs):
        if isinstance(fun, Callable):
            raise ValueError(f'Expected Callable, got {type(fun)}')

        if fun in self._require_adapt:
            fun_signature = signature(fun)
            def wrapped(*args, **kwargs):
                graph_arg = None
                adapted_graph = self._adapter.adapt(graph_arg)
                ...
        return fun


if __name__ == '__main__':
    pass
