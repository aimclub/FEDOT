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

    Optimiser operates with generic graph representation.
    Because of this any domain function requires adaptation
    of its graph arguments. Adapter can automatically adapt
    arguments to generic form in such cases.

    Important notions:
    - 'Domain' functions operate with domain-specific graphs.
    - 'Native' functions operate with generic graphs used by optimiser.
    - 'External' functions are functions defined by users of optimiser.
    (most notably, custom mutations and custom verifier rules).
    - 'Internal' functions are those defined by graph optimiser.
    (most notably, the default set of mutations and verifier rules).
    All internal functions are native.

    Adaptation registry usage and behavior:
    - Registry requires initialisation with specific adapter before usage.
    - Domain functions are adapted by default.
    - Native functions don't require adaptation of their arguments.
    - External functions are considered 'domain' functions by default.
    Hence, they're their arguments are adapted, unless users of optimiser
    exclude them from the process of automatic adaptation. It can be done
    by registering them as 'native'.
    """

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None):
        self.adapter = adapter or DirectAdapter()
        self._adaptee_cls = self.adapter._base_graph_class
        self._graph_cls = OptGraph
        self._native_opt_functions = set()

    def is_native(self, fun: Callable) -> bool:
        return fun in self._native_opt_functions

    def register_native(self, fun: Callable) -> Callable:
        """Registers callable object as an internal function that doesn't
        require adapt/restore mechanics when called inside the optimiser.
        Can be used as a decorator.

        :param fun: function or callable to be registered as native

        :return: same function without changes
        """
        self._native_opt_functions.add(fun)
        return fun

    def adapt(self, fun: Callable) -> Callable:
        """Adapts native function so that it could accept domain args.

        :param fun: native function that accepts native args (i.e. optimization graph)
         and requires adaptation of domain graph.

        :return: domain function that can be used inside Optimizer
        """
        return _transform(fun, f_args=self._maybe_adapt, f_ret=self._maybe_restore)

    def restore(self, fun: Callable) -> Callable:
        """Restores domain function so that it could accept native args,
        unless the function wasn't registered as native.

        :param fun: domain function that accepts domain args and required call to restore

        :return: native function that can be used inside Optimizer
        """
        if fun in self._native_opt_functions:
            return fun
        return _transform(fun, f_args=self._maybe_restore, f_ret=self._maybe_adapt)

    def _maybe_adapt(self, item):
        return self.adapter.adapt(item) if isinstance(item, self._adaptee_cls) else item

    def _maybe_restore(self, item):
        return self.adapter.restore(item) if isinstance(item, self._graph_cls) else item


def init_adapter(adapter: BaseOptimizationAdapter):
    AdaptRegistry(adapter)


def register_native(fun: Callable) -> Callable:
    return AdaptRegistry().register_native(fun)


def adapt(fun: Callable) -> Callable:
    return AdaptRegistry().adapt(fun)


def restore(fun: Callable) -> Callable:
    return AdaptRegistry().restore(fun)


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
