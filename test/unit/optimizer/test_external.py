import logging
from functools import partial
from typing import Optional, Union, Sequence

import pytest
from golem.core.dag.graph import Graph
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective.objective import Objective, ObjectiveFunction
from golem.core.optimisers.optimization_parameters import OptimizationParameters
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, AlgorithmParameters

from fedot.api.main import Fedot
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from test.integration.models.test_model import classification_dataset

_ = classification_dataset  # to avoid auto-removing of import


class StaticOptimizer(GraphOptimizer):
    """
    Dummy optimizer for testing
    """

    def __init__(self,
                 objective: Objective,
                 initial_graph: Union[Graph, Sequence[Graph]] = (),
                 requirements: Optional[OptimizationParameters] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_parameters: Optional[AlgorithmParameters] = None,
                 **kwargs):
        super().__init__(objective, initial_graph, requirements,
                         graph_generation_params, graph_optimizer_parameters)
        self.change_types = []
        self.node_name = kwargs.get('node_name') or 'rf'

    def optimise(self, objective: ObjectiveFunction):
        graph = OptGraph(OptNode(self.node_name))
        return [graph]


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_external_static_optimizer(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    automl = Fedot(problem='classification', timeout=0.2, logging_level=logging.DEBUG,
                   preset='fast_train',
                   with_tuning=False,
                   optimizer=partial(StaticOptimizer, node_name='logit'),
                   pop_size=2)
    obtained_pipeline = automl.fit(train_data)
    automl.predict(test_data)

    expected_pipeline = Pipeline(PipelineNode('logit'))

    assert obtained_pipeline.root_node.descriptive_id == expected_pipeline.root_node.descriptive_id
