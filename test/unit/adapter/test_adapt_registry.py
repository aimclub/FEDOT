from copy import deepcopy
from functools import partial

import pytest
from typing import Tuple

from fedot.core.adapter import *
from fedot.core.optimisers.fitness import Fitness, SingleObjFitness
from fedot.core.optimisers.graph import OptGraph, OptNode
from test.unit.adapter.mock_adapter import MockAdapter, MockDomainStructure
from test.unit.dag.test_graph_utils import graphs_same


@pytest.fixture(autouse=True)
def init_test_adapter():
    AdaptRegistry().init_adapter(MockAdapter())


@register_native
def decorated_native_func_add_node(graph: OptGraph):
    if not isinstance(graph, OptGraph):
        raise TypeError()
    new_graph = deepcopy(graph)
    new_graph.add_node(OptNode({'name': 'knn'}))
    return new_graph


def native_func_add_node(graph: OptGraph):
    if not isinstance(graph, OptGraph):
        raise TypeError()
    new_graph = deepcopy(graph)
    new_graph.add_node(OptNode({'name': 'scaling'}))
    return new_graph


def domain_func_add_node(struct: MockDomainStructure):
    if not isinstance(struct, MockDomainStructure):
        raise TypeError()
    new_struct = MockDomainStructure(struct.nodes)
    new_struct.nodes.append(OptNode({'name': 'knn'}))
    return new_struct


class CustomOperator:
    @register_native
    def decorated_native_func_add_node(self, graph: OptGraph):
        if not isinstance(graph, OptGraph):
            raise TypeError()
        new_graph = deepcopy(graph)
        new_graph.add_node(OptNode({'name': 'knn'}))
        return new_graph

    def native_func_add_node(self, graph: OptGraph):
        if not isinstance(graph, OptGraph):
            raise TypeError()
        new_graph = deepcopy(graph)
        new_graph.add_node(OptNode({'name': 'scaling'}))
        return new_graph

    def domain_func_add_node(self, struct: MockDomainStructure):
        if not isinstance(struct, MockDomainStructure):
            raise TypeError()
        new_struct = MockDomainStructure(struct.nodes)
        new_struct.nodes.append(OptNode({'name': 'knn'}))
        return new_struct


def domain_func_1arg(struct: MockDomainStructure):
    assert isinstance(struct, MockDomainStructure)


def domain_func_4args(arg1: int, struct1: MockDomainStructure, flag: bool, struct2: MockDomainStructure):
    assert isinstance(struct1, MockDomainStructure)
    assert isinstance(struct2, MockDomainStructure)


def domain_func_return1():
    return MockDomainStructure(nodes=[OptNode({'name': 'knn'})])


def domain_func_return3(struct: MockDomainStructure):
    some_flag = True
    return some_flag, struct, SingleObjFitness(0.5)


def domain_func_return_same(struct: MockDomainStructure):
    assert isinstance(struct, MockDomainStructure)
    return struct


def get_graphs() -> Tuple[OptGraph, MockDomainStructure]:
    opt_graph = OptGraph(OptNode({'name': 'knn'}, [OptNode({'name': 'scaling'})]))
    dom_struct = MockDomainStructure(opt_graph.nodes)
    return opt_graph, dom_struct


def test_adapt_1arg():
    opt_graph, dom_struct = get_graphs()

    func = domain_func_1arg
    restored_func = adapt(func)

    # test that opt graph is accepted by restored domain function
    restored_func(opt_graph)


def test_adapt_many_args():
    opt_graph, dom_struct = get_graphs()

    func = domain_func_4args
    restored_func = adapt(func)

    func(4, dom_struct, flag=True, struct2=dom_struct)
    # test that opt graph is accepted by restored domain function
    restored_func(4, opt_graph, flag=True, struct2=opt_graph)


def test_adapt_returned_same():
    opt_graph, dom_struct = get_graphs()

    func = domain_func_return_same
    restored_func = adapt(func)

    # sanity check
    returned_graph = func(dom_struct)
    assert returned_graph == dom_struct

    returned_graph = restored_func(opt_graph)
    assert graphs_same(returned_graph, opt_graph)
    # NB: identity of the graphs is not preserved
    assert id(returned_graph) != id(opt_graph)


def test_adapt_returned_single():
    func = domain_func_return1
    restored_func = adapt(func)

    return_value = restored_func()
    assert isinstance(return_value, OptGraph)


def test_adapt_returned_many():
    opt_graph, dom_struct = get_graphs()

    func = domain_func_return3
    restored_func = adapt(func)

    flag, graph, fitness = restored_func(opt_graph)
    # test that return value is adapted back to opt graph (if return is present)
    assert isinstance(graph, OptGraph)
    # and that other values are left unchanged
    assert isinstance(flag, bool)
    assert isinstance(fitness, Fitness)


@pytest.mark.parametrize('mutation', (
        decorated_native_func_add_node,
        domain_func_add_node,
        register_native(native_func_add_node),

        CustomOperator().decorated_native_func_add_node,
        CustomOperator().domain_func_add_node,
        register_native(CustomOperator().native_func_add_node),

        partial(decorated_native_func_add_node),
        partial(domain_func_add_node),
        partial(register_native(native_func_add_node)),
))
def test_adapt_registered_functions(mutation):
    """Demonstrates how both native & domain mutations are handled
    uniformly by adapter registry thanks to @register_native decorator."""

    opt_graph, dom_struct = get_graphs()

    restored_mutation = adapt(mutation)
    mutated_graph = restored_mutation(opt_graph)

    assert isinstance(mutated_graph, OptGraph)
    assert not graphs_same(mutated_graph, opt_graph)


@pytest.mark.parametrize('mutation', (
        native_func_add_node,
        CustomOperator().native_func_add_node,
        partial(native_func_add_node),
))
def test_adapt_unregistered_fail(mutation):
    opt_graph, dom_struct = get_graphs()

    was_registered = AdaptRegistry().is_native(mutation)
    if was_registered:
        AdaptRegistry().unregister_native(mutation)  # clear registration by previous tests

    restored_mutation = adapt(mutation)

    with pytest.raises(TypeError) as e:
        mutated_dom_graph = restored_mutation(dom_struct)
    with pytest.raises(TypeError) as e:
        mutated_graph = restored_mutation(opt_graph)

    if was_registered:
        AdaptRegistry().register_native(mutation)
