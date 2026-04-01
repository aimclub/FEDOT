import numpy as np

from fedot import Fedot
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class _TensorAutoPipeline:
    def __init__(self):
        self.preprocessor = 'pipeline-preprocessor'
        self.calls = []

    def unfit(self, unfit_preprocessor=True):
        self.calls.append(('unfit', unfit_preprocessor))

    def fit_tensordata(self, tensor_data, n_jobs=1):
        self.calls.append(('fit_tensordata', tensor_data, n_jobs))


class _StoredTrainData:
    task = Task(TaskTypesEnum.classification)
    data_type = DataTypesEnum.table
    target = np.array([0, 1])


def test_main_facade_fit_tensordata_auto_uses_benchmark_runtime_and_restores_tuning(monkeypatch):
    model = Fedot(
        problem='classification',
        with_tuning=True,
        benchmark_runtime_mode='tensor_gpu_bridge',
        benchmark_tensor_backend_name='gpu',
    )
    pipeline = _TensorAutoPipeline()
    stored_train_data = _StoredTrainData()
    captured = {}

    model.data_processor.to_input_data = lambda tensor_data: stored_train_data
    model.params.update_available_operations_by_preset = lambda train_data: captured.setdefault('updated_train_data', train_data)

    def fake_obtain_model(train_data, runtime_mode=None, tensor_backend_name=None):
        captured['train_data'] = train_data
        captured['runtime_mode'] = runtime_mode
        captured['tensor_backend_name'] = tensor_backend_name
        captured['with_tuning_during_obtain'] = model.params.get('with_tuning')
        return pipeline, [pipeline], 'history'

    monkeypatch.setattr(model.api_composer, 'obtain_model', fake_obtain_model)
    monkeypatch.setattr('fedot.api.main.BasePreprocessor.merge_preprocessors', lambda **kwargs: 'merged-preprocessor')
    monkeypatch.setattr('fedot.api.main.graph_structure', lambda pipeline_arg: 'tensor-auto-pipeline')

    result = model.fit_tensordata(tensor_data='tensor-data', predefined_model='auto')

    assert result is pipeline
    assert captured['train_data'] is stored_train_data
    assert captured['updated_train_data'] is stored_train_data
    assert captured['runtime_mode'] == 'tensor_gpu_bridge'
    assert captured['tensor_backend_name'] == 'gpu'
    assert captured['with_tuning_during_obtain'] is False
    assert model.params.get('with_tuning') is True
    assert pipeline.calls == [('unfit', False), ('fit_tensordata', 'tensor-data', model.params.n_jobs)]
    assert model.current_pipeline.preprocessor == 'merged-preprocessor'
    assert model.train_data is stored_train_data
    assert np.array_equal(model.target, stored_train_data.target)

