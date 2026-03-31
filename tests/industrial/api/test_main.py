from types import SimpleNamespace

import numpy as np

from fedot.core.data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.api.main import FedotIndustrial


class _DummyEncoder:
    def inverse_transform(self, values):
        return np.array(values) + 10

    def transform(self, values):
        return np.array(values) + 1


def test_industrial_main_abstract_predict_uses_rule_based_pipeline_path(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    predict_data = SimpleNamespace(
        task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
        target=np.array([1, 2, 3, 4]),
    )
    industrial.predict_data = predict_data
    industrial.target_encoder = _DummyEncoder()
    industrial.manager = SimpleNamespace(
        solver=SimpleNamespace(predict=lambda data, mode: OutputData(
            idx=np.arange(4),
            predict=np.array([1, 2, 3, 4]),
            target=None,
            task=Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=2)),
            data_type=DataTypesEnum.ts,
        )),
        condition_check=SimpleNamespace(
            solver_have_target_encoder=lambda encoder: True,
            solver_is_fedot_class=lambda solver: False,
            solver_is_pipeline_class=lambda solver: True,
        ),
    )

    result = industrial._FedotIndustrial__abstract_predict(predict_data, 'labels')

    assert np.array_equal(result, np.array([13, 14]))
    assert np.array_equal(industrial.predict_data.target, np.array([11, 12, 13, 14]))


def test_industrial_main_metric_evaluation_loop_uses_rule_based_shape_and_encoder(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    industrial.target_encoder = _DummyEncoder()
    industrial.manager = SimpleNamespace(
        condition_check=SimpleNamespace(solver_have_target_encoder=lambda encoder: True),
    )
    captured = {}

    monkeypatch.setattr(
        'fedot.industrial.api.main.FEDOT_GET_METRICS',
        {'classification': lambda **kwargs: captured.update(kwargs) or {'f1': 0.8}},
    )

    result = industrial._metric_evaluation_loop(
        target=np.array([[0], [1]]),
        predicted_labels=np.array([1, 0]),
        predicted_probs=np.array([[0.3, 0.7], [0.6, 0.4]]),
        problem='classification',
        metric_names=('f1',),
        rounding_order=3,
        train_data='train-data',
        seasonality=1,
    )

    assert result == {'f1': 0.8}
    assert np.array_equal(captured['target'], np.array([1, 2]))
    assert np.array_equal(captured['labels'], np.array([[2], [1]]))
    assert captured['metric_names'] == ('f1',)
    assert captured['train_data'] == 'train-data'



def test_industrial_main_explain_uses_rule_based_config():
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = {}

    class FakeExplainer:
        def __init__(self, model, features, target):
            captured['features'] = features
            captured['target'] = target

        def explain(self, n_samples, window, method):
            captured['explain'] = (n_samples, window, method)

        def visual(self, metric, threshold, name):
            captured['visual'] = (metric, threshold, name)

    industrial.manager = SimpleNamespace(
        industrial_config=SimpleNamespace(explain_methods={'recurrence': FakeExplainer}),
        predict_data=SimpleNamespace(features=np.arange(6).reshape(1, 2, 3), target=np.array([1, 2])),
    )

    industrial.explain({'method': 'recurrence', 'samples': 3, 'window': 7, 'metric': 'f1', 'threshold': 80, 'name': 'demo'})

    assert captured['features'].shape == (2, 3)
    assert np.array_equal(captured['target'], np.array([1, 2]))
    assert captured['explain'] == (3, 7, 'f1')
    assert captured['visual'] == ('f1', 80, 'demo')


def test_industrial_main_load_uses_rule_based_path_resolution(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    industrial.manager = SimpleNamespace()

    monkeypatch.setattr('fedot.industrial.api.main.IndustrialModels', lambda: SimpleNamespace(setup_repository=lambda backend=None: 'repo'))
    monkeypatch.setattr('fedot.industrial.api.main.os.listdir', lambda path: ['pipeline_saved_1', 'fitted_operations'])
    monkeypatch.setattr('fedot.industrial.api.main.Pipeline', lambda: SimpleNamespace(load=lambda path: f'loaded:{path}'))

    result = industrial.load('root')

    assert result == ['loaded:root/pipeline_saved_1/pipeline_saved_1/0_pipeline_saved',
                      'loaded:root/pipeline_saved_1/fitted_operations/0_pipeline_saved']


def test_industrial_main_save_uses_rule_based_mode_selection(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = []

    class FakeManager:
        def __init__(self):
            self.solver = SimpleNamespace(current_pipeline=SimpleNamespace(save=lambda **kwargs: captured.append(('model', kwargs))))
            self.compute_config = SimpleNamespace(output_folder='out')
            self.condition_check = SimpleNamespace(solver_is_fedot_class=lambda solver: False)
            self.logger = SimpleNamespace(info=lambda msg: captured.append(('log', msg)))
            self.predicted_labels = np.array([1, 0])

        def create_folder(self, output_folder):
            captured.append(('folder', output_folder))

    industrial.manager = FakeManager()
    industrial.metric_dict = SimpleNamespace(to_csv=lambda path: captured.append(('metrics', path)))

    monkeypatch.setattr('fedot.industrial.api.main.pd.DataFrame', lambda values: SimpleNamespace(to_csv=lambda path: captured.append(('prediction', path))))

    industrial.save(mode='prediction')

    assert ('folder', 'out') in captured
    assert ('prediction', 'out/labels.csv') in captured


def test_industrial_main_vis_history_uses_rule_based_mode(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = []

    class FakeHistoryVisualizer:
        def __init__(self, history):
            self.history = 'history-object'

        def fitness_box(self, **kwargs):
            captured.append(('fitness', kwargs))

        def operations_animated_bar(self, **kwargs):
            captured.append(('models', kwargs))

        def diversity_population(self, **kwargs):
            captured.append(('diversity', kwargs))

    monkeypatch.setattr('fedot.industrial.api.main.OptHistory.load', lambda path: 'loaded-history')
    monkeypatch.setattr('fedot.industrial.api.main.PipelineHistoryVisualizer', FakeHistoryVisualizer)

    result = industrial.vis_optimisation_history(opt_history_path='history', mode='models', return_history=True)

    assert result == 'history-object'
    assert captured == [('models', {'save_path': 'operations_animated_bar.gif', 'show_fitness': True})]
