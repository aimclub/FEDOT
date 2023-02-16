import logging
from typing import Any, Optional, Union, Sequence

import pytest

from fedot.api.main import Fedot
from fedot.core.dag.graph import Graph
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, GraphOptimizerParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.optimisers.objective.objective import Objective, ObjectiveFunction
from test.unit.models.test_model import classification_dataset

_ = classification_dataset  # to avoid auto-removing of import


class StaticOptimizer(GraphOptimizer):
    """
    Dummy optimizer for testing
    """

    def __init__(self,
                 objective: Objective,
                 initial_graph: Union[Graph, Sequence[Graph]] = (),
                 requirements: Optional[Any] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_parameters: Optional[GraphOptimizerParameters] = None,
                 **kwargs):
        super().__init__(objective, initial_graph, requirements,
                         graph_generation_params, graph_optimizer_parameters)
        self.change_types = []
        self.node_name = kwargs.get('node_name') or 'logit'

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
                   optimizer=StaticOptimizer,
                   pop_size=2,
                   optimizer_external_params={'node_name': 'logit'})
    obtained_pipeline = automl.fit(train_data)
    automl.predict(test_data)

    expected_pipeline = Pipeline(PipelineNode('logit'))

    assert obtained_pipeline.root_node.descriptive_id == expected_pipeline.root_node.descriptive_id
