from inspect import signature, Parameter
from typing import Optional, Callable, Any

from fedot.core.dag.graph import Graph
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

    __adapt_target_cls = Graph

    def __init__(self, adapter: Optional[BaseOptimizationAdapter] = None):
        self._adapter = adapter or DirectAdapter()
        self._adaptee_cls = self._adapter._base_graph_class
        self._graph_cls = OptGraph
        # self._require_adapt = {}
        # self._require_restore = {}

    @staticmethod
    def adapt2(fun: Callable):
        return AdaptRegistry().adapt(fun)

    def adapt(self, fun: Callable):
        if not isinstance(fun, Callable):
            raise ValueError(f'Expected Callable, got {type(fun)}')

        # if fun not in self._require_adapt:
        if False:
            adapted_fun = fun
        else:
            def adapted_fun(*args, **kwargs):
                adapted_args = (self._maybe_adapt(arg) for arg in args)
                adapted_kwargs = dict((kw, self._maybe_adapt(arg)) for kw, arg in kwargs)
                result = fun(*adapted_args, **adapted_kwargs)
                return self._maybe_restore(result)

        return adapted_fun

    def _maybe_adapt(self, item):
        return self._adapter.adapt(item) if isinstance(item, self._adaptee_cls) else item

    def _maybe_restore(self, item):
        return self._adapter.restore(item) if isinstance(item, self._graph_cls) else item


class TestCls(metaclass=SingletonMeta):
    def __init__(self, prefix: str = ''):
        self.prefix = prefix

    def __str__(self):
        return self.prefix

    @staticmethod
    def decorate(fun):
        def decorated(*args, **kwargs):
            print(f'decorated with {TestCls.__name__}')
            return fun(*args, **kwargs)
        return decorated

    @staticmethod
    def decorate2(fun):
        return TestCls().decorate_local(fun)

    def decorate_local(self, fun):
        def decorated(*args, **kwargs):
            print(f'{self.prefix}: decorated with {id(self)} of {self.__class__.__name__}')
            return fun(*args, **kwargs)
        return decorated


decorate_cls = TestCls('global blah!')


# @TestCls.decorate
# @decorate_cls.decorate_local
# @TestCls().decorate_local
@TestCls.decorate2
def tst_function(x: int, s: str = None):
    return x*x


if __name__ == '__main__':
    print('str: ' + str(TestCls('local blah!')))
    TestCls().prefix = 'subst reblah!!'
    print(tst_function(5))
