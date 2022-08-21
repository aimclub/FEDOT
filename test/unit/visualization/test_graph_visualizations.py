from pathlib import Path
from typing import Type, Union

import pytest

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_delegate import GraphDelegate
from fedot.core.dag.graph_node import GraphNode
from fedot.core.dag.graph_operator import GraphOperator
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.pipelines.node import Node
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import DEFAULT_PARAMS_STUB


@pytest.fixture(scope='module', params=[GraphDelegate, GraphOperator, Pipeline, OptGraph])
def graph(request):
    graph_type: Union[Type[Graph], Type[Pipeline], Type[OptGraph]] = request.param
    nodes_kwargs = [{'content': {'name': f'n{i+1}', 'params': DEFAULT_PARAMS_STUB}} for i in range(4)]
    nodes_kwargs[-1]['nodes_from'] = range(len(nodes_kwargs) - 1)
    if graph_type in [GraphOperator, GraphDelegate]:
        node_type = GraphNode
    elif graph_type is Pipeline:
        node_type = Node
    else:
        node_type = OptNode
    nodes = []
    for i, kwargs in enumerate(nodes_kwargs):
        if 'nodes_from' in kwargs:
            kwargs['nodes_from'] = [nodes[j] for j in kwargs['nodes_from']]
        else:
            kwargs['nodes_from'] = []
        nodes.append(node_type(**kwargs))
    return graph_type(nodes[-1])


@pytest.mark.parametrize('engine', ('matplotlib', 'pyvis', 'graphviz'))
def test_graph_show_saving_plots(graph, engine, tmp_path):
    save_path = Path(tmp_path, engine)
    save_path = save_path.with_suffix('.html') if engine == 'pyvis' else save_path.with_suffix('.png')
    try:
        graph.show(engine=engine, save_path=save_path, dpi=100)
        assert save_path.exists()
    except ImportError:
        assert engine == 'graphviz'
