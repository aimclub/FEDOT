from inspect import signature, Parameter
from typing import Optional, Callable, Any, Tuple

from fedot.core.adapter.adapter import BaseOptimizationAdapter, DirectAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.utilities.singleton_meta import SingletonMeta


# TODO: tests for
#  - graph argument in different places .
#  - no graph argument
#  - several graph arguments
#  - already adapted graph arg
# TODO: tests for
#  - decorator before init; with normal call
#  - decorator before init; with too-early call

# TODO: registration of all native functions

class AdaptRegistry(metaclass=SingletonMeta):
    """Registry of functions that require adapter.
    Adaptation Registry enables automatic transformation
    between internal and domain graph representations.

    Singleton; requires initialisation with specific adapter before usage.

    Aimed at usage inside the context of Optimiser that internally
    operates with generic graph representation. Because of this
    any domain function requires adaptation of its arguments.

    'Domain' functions operate with domain-specific graphs.
    'Native' functions operate with generic graphs used by optimiser.
    'External' functions are functions defined by users of optimiser.
    (most notably, custom mutations and custom verifier rules).
    'Internal' functions are those defined by graph optimiser.
    (most notably, the default set of mutations and verifier rules).

    All internal functions are native. Native functions avoid automatic
    adaptation, while arguments to domain functions are adapted by default.
    Users of optimiser can register external functions as 'native'
    to exclude them from the process of automatic adaptation.
    """

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None):
        self.adapter = adapter or DirectAdapter()
        self._adaptee_cls = self.adapter._base_graph_class
        self._graph_cls = OptGraph
        self._native_opt_functions = set()

    def is_native(self, fun: Callable):
        return fun in self._native_opt_functions

    def register_native(self, fun: Callable):
        """Registers callable object as an internal function that doesn't
        require adapt/restore mechanics when called inside the optimiser.
        Can be used as a decorator."""
        self._native_opt_functions.add(fun)
        return fun

    def adapt(self, fun: Callable):
        return _transform(fun, self._maybe_adapt, self._maybe_restore)

    def restore(self, fun: Callable):
        """Conditionally restores function if it wasn't registered as native."""
        if fun in self._native_opt_functions:
            return fun
        return _transform(fun, self._maybe_restore, self._maybe_adapt)

    def _maybe_adapt(self, item):
        return self.adapter.adapt(item) if isinstance(item, self._adaptee_cls) else item

    def _maybe_restore(self, item):
        return self.adapter.restore(item) if isinstance(item, self._graph_cls) else item


def init_adapter(adapter: BaseOptimizationAdapter):
    AdaptRegistry(adapter)


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

        if isinstance(result, Tuple):
            # In case when function returns not only Graph
            adapted_result = (f_ret(result_item) for result_item in result)
        else:
            adapted_result = f_ret(result)
        return adapted_result

    return adapted_fun
