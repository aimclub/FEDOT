from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.repository.tasks import Task, TaskTypesEnum


class _FakeTuner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_tuner_builder_build_uses_regular_splitter_boundary(monkeypatch):
    captured = {}

    class FakeSplitter:
        def __init__(self, cv_folds, validation_blocks=None):
            captured['splitter_init'] = (cv_folds, validation_blocks)
            self.validation_blocks = 7

        def build(self, data):
            captured['build_data'] = data
            return 'regular-producer'

    monkeypatch.setattr('fedot.core.pipelines.tuning.tuner_builder.DataSourceSplitter', FakeSplitter)
    monkeypatch.setattr(
        'fedot.core.pipelines.tuning.tuner_builder.MetricsObjective',
        lambda metric, is_multi_objective=False: ('objective', metric, is_multi_objective),
    )

    def fake_objective_evaluate(objective, data_producer, time_constraint=None, eval_n_jobs=1, validation_blocks=None):
        captured['objective'] = objective
        captured['data_producer'] = data_producer
        captured['validation_blocks'] = validation_blocks
        return 'objective-evaluate'

    monkeypatch.setattr(
        'fedot.core.pipelines.tuning.tuner_builder.PipelineObjectiveEvaluate',
        fake_objective_evaluate,
    )

    builder = TunerBuilder(Task(TaskTypesEnum.classification))
    builder.tuner_class = _FakeTuner

    tuner = builder.build('input-data')

    assert captured['build_data'] == 'input-data'
    assert captured['data_producer'] == 'regular-producer'
    assert captured['validation_blocks'] == 7
    assert tuner.kwargs['objective_evaluate'] == 'objective-evaluate'


def test_tuner_builder_build_tensordata_uses_tensor_splitter_boundary(monkeypatch):
    captured = {}

    class FakeSplitter:
        def __init__(self, cv_folds, validation_blocks=None):
            captured['splitter_init'] = (cv_folds, validation_blocks)
            self.validation_blocks = 5

        def build_tensordata(self, tensor_data):
            captured['tensor_data'] = tensor_data
            return 'tensor-producer'

    monkeypatch.setattr('fedot.core.pipelines.tuning.tuner_builder.DataSourceSplitter', FakeSplitter)
    monkeypatch.setattr(
        'fedot.core.pipelines.tuning.tuner_builder.MetricsObjective',
        lambda metric, is_multi_objective=False: ('objective', metric, is_multi_objective),
    )

    def fake_objective_evaluate(objective, data_producer, time_constraint=None, eval_n_jobs=1, validation_blocks=None):
        captured['objective'] = objective
        captured['data_producer'] = data_producer
        captured['validation_blocks'] = validation_blocks
        return 'objective-evaluate'

    monkeypatch.setattr(
        'fedot.core.pipelines.tuning.tuner_builder.PipelineObjectiveEvaluate',
        fake_objective_evaluate,
    )

    builder = TunerBuilder(Task(TaskTypesEnum.classification))
    builder.tuner_class = _FakeTuner

    tuner = builder.build_tensordata('tensor-data')

    assert captured['tensor_data'] == 'tensor-data'
    assert captured['data_producer'] == 'tensor-producer'
    assert captured['validation_blocks'] == 5
    assert tuner.kwargs['objective_evaluate'] == 'objective-evaluate'
