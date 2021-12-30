import json
import os

from fedot.api.main import Fedot
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.opt_history import ParentOperator
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.serializers import Serializer
from fedot.core.utils import fedot_project_root


def test_parent_operator():
    pipeline = Pipeline(PrimaryNode('linear'))
    adapter = PipelineAdapter()
    ind = Individual(adapter.adapt(pipeline))
    ind.graph.uid = pipeline.uid
    mutation_type = MutationTypesEnum.simple

    operator_for_history = ParentOperator(operator_type='mutation',
                                          operator_name=str(mutation_type),
                                          parent_objects=[ind])

    assert operator_for_history.parent_objects[0].graph.uid == pipeline.uid
    assert operator_for_history.operator_type == 'mutation'


def test_operators_in_history():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')

    auto_model = Fedot(problem='classification', seed=42,
                       timeout=0.1,
                       composer_params={'num_of_generations': 3, 'pop_size': 4},
                       preset='fast_train')
    auto_model.fit(features=file_path_train, target='Y')

    assert auto_model.history is not None
    assert 1 <= len(auto_model.history.individuals) <= 3

    # test history dumps
    dumped_history = auto_model.history.save()

    assert dumped_history is not None
