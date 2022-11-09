from copy import copy
from functools import partial
from typing import Callable

from fedot.core.utilities.singleton_meta import SingletonMeta


class AdaptRegistry(metaclass=SingletonMeta):
    """Registry of callables that require adaptation of argument/return values.
    AdaptRegistry together with :py:class:`fedot.core.adapter.adapter.BaseOptimizationAdapter`
    enables automatic transformation between internal and domain graph representations.

    **Short description of the use-case**

    Operators & verification rules that operate on internal representation
    of graphs must be marked as native with decorator
    :py:func:`fedot.core.adapter.adapt_registry.register_native`.

    Usually this is the case when users of the framework provide custom
    operators for internal optimization graphs. When custom operators
    operate on domain graphs, nothing is required.

    **Extended description**

    Optimiser operates with generic graph representation.
    Because of this any domain function requires adaptation
    of its graph arguments. Adapter can automatically adapt
    arguments to generic form in such cases.

    Important notions:

    * 'Domain' functions operate with domain-specific graphs.
    * 'Native' functions operate with generic graphs used by optimiser.
    * 'External' functions are functions defined by users of optimiser.

        Most notably, custom mutations and custom verifier rules.

    * 'Internal' functions are those defined by graph optimiser.

        Most notably, the default set of mutations and verifier rules.
        All internal functions are native.

    Adaptation registry usage and behavior:

    * Domain functions are adapted by default.
    * Native functions don't require adaptation of their arguments.
    * External functions are considered 'domain' functions by default.

        Hence, their arguments are adapted, unless users of optimiser
        exclude them from the process of automatic adaptation.
        It can be done by registering them as 'native'.

    AdaptRegistry can be safely used with multiprocessing
    insofar as all relevant functions are registered as native
    in the main process before child processes are started.
    """

    _native_flag_attr_name_ = '_fedot_is_optimizer_native'

    def __init__(self):
        self._registered_native_callables = []

    def register_native(self, fun: Callable) -> Callable:
        """Registers callable object as an internal function
        that can work with internal graph representation.
        Hence, it doesn't require adaptation when called by the optimiser.

        Implementation details.
        Works by setting a special attribute on the object.
        This attribute then is checked by ``is_native`` used by adapters.

        Args:
            fun: function or callable to be registered as native

        Returns:
            Callable: same function with special private attribute set
        """
        original_function = AdaptRegistry._get_underlying_func(fun)
        setattr(original_function, AdaptRegistry._native_flag_attr_name_, True)
        self._registered_native_callables.append(original_function)
        return fun

    def unregister_native(self, fun: Callable) -> Callable:
        """Unregisters callable object. See ``register_native``.

        Args:
            fun: function or callable to be unregistered as native

        Returns:
            Callable: same function with special private attribute unset
        """
        original_function = AdaptRegistry._get_underlying_func(fun)
        if hasattr(original_function, AdaptRegistry._native_flag_attr_name_):
            delattr(original_function, AdaptRegistry._native_flag_attr_name_)
            self._registered_native_callables.remove(original_function)
        return fun

    @staticmethod
    def is_native(fun: Callable) -> bool:
        """Tests callable object for a presence of specific attribute
        that tells that this function must not be restored with Adapter.

        Args:
            fun: tested Callable (function, method, functools.partial, or any callable object)

        Returns:
            bool: True if the callable was registered as native, False otherwise.
        """
        original_function = AdaptRegistry._get_underlying_func(fun)
        is_native = getattr(original_function, AdaptRegistry._native_flag_attr_name_, False)
        return is_native

    def clear_registered_callables(self):
        # copy is to avoid removing elements from list while iterating
        for f in copy(self._registered_native_callables):
            self.unregister_native(f)

    @staticmethod
    def _get_underlying_func(obj: Callable) -> Callable:
        """Recursively unpacks 'partial' and 'method' objects to get underlying function.

        Args:
            obj: callable to try unpacking

        Returns:
            Callable: unpacked function that underlies the callable, or the unchanged object itself
        """
        while True:
            if isinstance(obj, partial):  # if it is a 'partial'
                obj = obj.func
            elif hasattr(obj, '__func__'):  # if it is a 'method'
                obj = obj.__func__
            else:
                return obj  # return the unpacked underlying function or the original object


def register_native(fun: Callable) -> Callable:
    """Out-of-class version of the ``register_native``
    function that's intended to be used as a decorator.

    Args:
        fun: function or callable to be registered as native

    Returns:
        Callable: same function with special private attribute set
    """
    return AdaptRegistry().register_native(fun)
