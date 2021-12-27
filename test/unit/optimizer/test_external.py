from typing import Any, Callable, List, Optional, Union

import pytest

from fedot.api.main import Fedot
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import Log
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams, GraphOptimiser, GraphOptimiserParameters
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import (MetricsEnum)
from test.unit.models.test_model import classification_dataset

_ = classification_dataset  # to avoid auto-removing of import


class StaticOptimizer(GraphOptimiser):
    """
    Dummy optimizer for testing
    """

    def __init__(self, initial_graph: Union[Any, List[Any]],
                 requirements: Any,
                 graph_generation_params: GraphGenerationParams,
                 metrics: List[MetricsEnum],
                 parameters: GraphOptimiserParameters = None,
                 log: Log = None):
        super().__init__(initial_graph, requirements, graph_generation_params, metrics, parameters, log)
        self.change_types = []

    def optimise(self, objective_function,
                 on_next_iteration_callback: Optional[Callable] = None,
                 show_progress: bool = True):
        return OptGraph(OptNode('xgboost'))


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_external_static_optimizer(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    automl = Fedot(problem='classification', timeout=1, verbose_level=4,
                   preset='light_notun')
    automl.api_composer.optimiser = StaticOptimizer

    obtained_pipeline = automl.fit(train_data)
    automl.predict(test_data)

    expected_pipeline = Pipeline(PrimaryNode('xgboost'))

    assert obtained_pipeline.root_node.descriptive_id == expected_pipeline.root_node.descriptive_id
