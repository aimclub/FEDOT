from inspect import signature, Parameter
from typing import Optional, Callable, Any

from fedot.core.dag.graph import Graph
from fedot.core.optimisers.adapters import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.utilities.singleton import SingletonMeta


class AdaptRegistry(metaclass=SingletonMeta):

    __adapt_target_cls = Graph

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None):
        self._adapter = adapter or DirectAdapter()
        self._adaptee_cls = self._adapter._base_graph_class
        self._graph_cls = OptGraph
        self._require_adapt = {}
        self._require_restore = {}

    def adapt(self, fun: Callable[..., Any], *args, **kwargs):
        if isinstance(fun, Callable):
            raise ValueError(f'Expected Callable, got {type(fun)}')

        # TODO: tests for
        #  - graph argument in different places .
        #  - no graph argument
        #  - several graph arguments
        #  - already adapted graph arg
        # TODO: tests for
        #  - decorator before init; with normal call
        #  - decorator before init; with too-early call

        if fun in self._require_adapt:
            fun_signature = signature(fun)
            graph_arg = None
            # NB: only positional annotated graph arguments can be adapted
            adapted_args_positions = []
            adapted_args_names = []
            for i, (name, param) in enumerate(fun_signature.parameters):
                param: Parameter
                ann = param.annotation
                requires_adapt = (ann in (Graph, 'Graph'))
                if requires_adapt:
                    adapted_args_positions.append(i)
                    adapted_args_names.append(name)

            def wrapped(*args, **kwargs):
                adapted_args = (self._maybe_adapt(arg) for arg in args)
                adapted_kwargs = dict((kw, self._maybe_adapt(arg)) for kw, arg in kwargs)

                result = fun(*adapted_args, **adapted_kwargs)

                return self._maybe_restore(result)

        return wrapped

    def _maybe_adapt(self, item):
        return self._adapter.adapt(item) if isinstance(item, self._adaptee_cls) else item

    def _maybe_restore(self, item):
        return self._adapter.restore(item) if isinstance(item, self._graph_cls) else item


if __name__ == '__main__':
    pass
