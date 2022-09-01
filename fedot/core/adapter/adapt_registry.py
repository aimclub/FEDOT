from functools import partial
from typing import Optional, Callable, Any, Tuple

from fedot.core.adapter.adapter import BaseOptimizationAdapter, DirectAdapter
from fedot.core.dag.graph import Graph
from fedot.core.optimisers.graph import OptGraph
from fedot.core.utilities.singleton_meta import SingletonMeta


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

    _native_flag_attr_name_ = '_fedot_is_optimizer_native'

    def __init__(self):
        self.adapter = DirectAdapter(Graph)
        self._domain_struct_cls = Graph
        self._opt_graph_cls = OptGraph
        self._registered_native_callables = []

    def init_adapter(self, adapter: BaseOptimizationAdapter):
        """Initialize Adapter Registry with specific Adapter"""
        self.adapter = adapter
        self._domain_struct_cls = self.adapter.domain_graph_class

    def register_native(self, fun: Callable) -> Callable:
        """Registers callable object as an internal function that doesn't
        require adapt/restore mechanics when called inside the optimiser.
        Allows callable to receive non-adapted OptGraph used by the optimiser.

        :param fun: function or callable to be registered as native

        :return: same function with special private attribute set
        """
        original_function = AdaptRegistry._get_underlying_func(fun)
        setattr(original_function, AdaptRegistry._native_flag_attr_name_, True)
        self._registered_native_callables.append(original_function)
        return fun

    def unregister_native(self, fun: Callable) -> Callable:
        """Unregisters callable object. See ``register_native``."""
        original_function = AdaptRegistry._get_underlying_func(fun)
        if hasattr(original_function, AdaptRegistry._native_flag_attr_name_):
            delattr(original_function, AdaptRegistry._native_flag_attr_name_)
        self._registered_native_callables.remove(original_function)
        return fun

    @staticmethod
    def is_native(fun: Callable) -> bool:
        """Tests callable object for a presence of specific attribute
        that tells that this function must not be restored with Adapter.

        :param fun: tested Callable (function, method, functools.partial, or any callable object)
        :return: True if the callable was registered as native, False otherwise."""

        original_function = AdaptRegistry._get_underlying_func(fun)
        is_native = getattr(original_function, AdaptRegistry._native_flag_attr_name_, False)
        return is_native

    def clear_registered_callables(self):
        for f in self._registered_native_callables:
            self.unregister_native(f)

    def restore(self, fun: Callable) -> Callable:
        """Wraps native function so that it could accept domain graphs as arguments.

        Behavior: `restore( f(OptGraph) ) => f'(DomainGraph)`

        :param fun: native function that accepts native args (i.e. optimization graph)
         and requires adaptation of domain graph.

        :return: domain function that can be used inside Optimizer
        """
        return _transform(fun, f_args=self._maybe_adapt, f_ret=self._maybe_restore)

    def adapt(self, fun: Callable) -> Callable:
        """Wraps domain function so that it could accept native optimization graphs
        as arguments. If the function was registered as native, it is returned as-is.

        Behavior: `adapt( f(DomainGraph) ) => f'(OptGraph)`

        :param fun: domain function that accepts domain args and required call to restore

        :return: native function that can be used inside Optimizer
        """
        if AdaptRegistry.is_native(fun):
            return fun
        return _transform(fun, f_args=self._maybe_restore, f_ret=self._maybe_adapt)

    @staticmethod
    def _get_underlying_func(obj: Callable) -> Callable:
        """Recursively unpacks 'partial' and 'method' objects to get underlying function.

        :param obj: callable to try unpacking
        :return: unpacked function that underlies the callable, or the unchanged object itself
        """
        while True:
            if isinstance(obj, partial):  # if it is a 'partial'
                obj = obj.func
            elif hasattr(obj, '__func__'):  # if it is a 'method'
                obj = obj.__func__
            else:
                return obj  # return unpacked the underlying function or original object

    def _maybe_adapt(self, item):
        return self.adapter.adapt(item) if isinstance(item, self._domain_struct_cls) else item

    def _maybe_restore(self, item):
        return self.adapter.restore(item) if isinstance(item, self._opt_graph_cls) else item


def register_native(fun: Callable) -> Callable:
    """Out-of-class version of the function intended to be used as decorator."""
    return AdaptRegistry().register_native(fun)


def restore(fun: Callable) -> Callable:
    """Out-of-class version of the function intended to be used as decorator."""
    return AdaptRegistry().restore(fun)


def adapt(fun: Callable) -> Callable:
    """Out-of-class version of the function intended to be used as decorator."""
    return AdaptRegistry().adapt(fun)


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
