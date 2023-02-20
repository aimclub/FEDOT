import json
import uuid
from datetime import timedelta
from typing import Tuple, Union, Optional, Sequence

import pytest
from golem.core.utilities.serializable import Serializable

from fedot.remote.infrastructure.clients.client import Client
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams


class MockSerializableGraph(Serializable):

    def __init__(self, evaluated: bool = False):
        self.evaluated = evaluated
        self.id = str(uuid.uuid4())

    def evaluate(self):
        self.evaluated = True

    def save(self, path: str = None, datetime_in_path: bool = True) -> Tuple[str, dict]:
        data = {'evaluated': self.evaluated, 'id': self.id}
        saved = json.dumps(data)
        return saved, {}

    def load(self, source: Union[str, dict], internal_state_data: Optional[dict] = None):
        data = json.loads(source)
        self.__dict__.update(data)


class TestLocalClient(Client):
    def __init__(self):
        self.graphs = []
        super().__init__(connect_params={}, exec_params={}, output_path=None)

    def create_task(self, config):
        # here config is just a json dump of the graph
        graph = MockSerializableGraph.from_serialized(config)
        graph.evaluate()
        dumped = graph.save()

        task_id = str(len(self.graphs))
        self.graphs.append(dumped)
        return task_id

    def wait_until_ready(self) -> timedelta:
        return timedelta()

    def download_result(self, execution_id: str, result_cls=MockSerializableGraph) -> MockSerializableGraph:
        index = int(execution_id)
        graph_json, additional_data = self.graphs[index]
        graph = MockSerializableGraph.from_serialized(graph_json, additional_data)
        return graph


def mock_config(graph_json, *args, **kwargs):
    return graph_json


@pytest.fixture(autouse=True)
def init_remote_evaluator():
    # runs around tests thanks to 'yield'

    evaluator = RemoteEvaluator()
    evaluator.init(TestLocalClient(), RemoteTaskParams(mode='remote'), mock_config)

    # marks the ned of test startup and beginning of test shutdown
    yield

    # return evaluator to local mode
    evaluator = RemoteEvaluator()
    evaluator.init(None, RemoteTaskParams(mode='local'))


def get_many_graphs(number: int = 1) -> Sequence[MockSerializableGraph]:
    graphs = []
    for i in range(number):
        graphs.append(MockSerializableGraph())
    return graphs


@pytest.mark.parametrize('num_of_graphs', [0, 1, 2, 10])
def test_remote_composer_custom_graph(num_of_graphs):
    graphs = get_many_graphs(num_of_graphs)

    assert not any(graph.evaluated for graph in graphs)

    evaluated_graphs = RemoteEvaluator().compute_graphs(graphs)

    assert len(graphs) == len(evaluated_graphs)
    assert all(graph.evaluated for graph in evaluated_graphs)
