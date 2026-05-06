from types import SimpleNamespace

import numpy as np

from fedot.core.data.input_data.data import OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.api.main import FedotIndustrial
from fedot.industrial.core.repository.constanst_repository import FEDOT_TUNER_STRATEGY, FEDOT_TUNING_METRICS


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


def test_industrial_main_fit_uses_rule_based_solver_path(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = {}
    industrial.manager = SimpleNamespace(
        industrial_config=SimpleNamespace(strategy=object()),
        solver=SimpleNamespace(fit=lambda data: captured.update(path='solver', data=data)),
    )
    industrial._process_input_data = lambda data: 'processed-train'
    industrial._FedotIndustrial__init_industrial_backend = lambda data: data
    industrial._FedotIndustrial__init_solver = lambda data: data

    industrial.fit(('features', 'target'))

    assert captured == {'path': 'solver', 'data': 'processed-train'}


def test_industrial_main_fit_uses_rule_based_strategy_path(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = {}

    class _CallableStrategy:
        def __call__(self):
            return None

        def fit(self, data):
            captured.update(path='strategy', data=data)

    industrial.manager = SimpleNamespace(
        industrial_config=SimpleNamespace(strategy=_CallableStrategy()),
        solver=SimpleNamespace(fit=lambda data: captured.update(path='solver', data=data)),
    )
    industrial._process_input_data = lambda data: 'processed-train'
    industrial._FedotIndustrial__init_industrial_backend = lambda data: data
    industrial._FedotIndustrial__init_solver = lambda data: data

    industrial.fit(('features', 'target'))

    assert captured == {'path': 'strategy', 'data': 'processed-train'}


def test_industrial_main_predict_proba_uses_rule_based_mode_normalization(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = {}
    industrial.manager = SimpleNamespace(
        compute_config=SimpleNamespace(backend='cpu'),
        industrial_config=SimpleNamespace(is_regression_task_context=True),
    )
    industrial._process_input_data = lambda data: 'processed-predict'
    industrial._FedotIndustrial__abstract_predict = lambda data, mode: captured.update(
        data=data, mode=mode) or 'predicted-probs'

    monkeypatch.setattr(
        'fedot.industrial.api.main.IndustrialModels',
        lambda: SimpleNamespace(
            setup_repository=lambda backend=None: f'repo:{backend}'))

    result = industrial.predict_proba(('features', 'target'), predict_mode='probs')

    assert result == 'predicted-probs'
    assert captured == {'data': 'processed-predict', 'mode': 'labels'}
    assert industrial.manager.predicted_probs == 'predicted-probs'


def test_industrial_main_finetune_uses_rule_based_tuning_plan(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = {}

    class _BuiltModel:
        def fit(self, data):
            captured['fitted_data'] = data

    class _ModelToTune:
        def build(self):
            return _BuiltModel()

    industrial.manager = SimpleNamespace(
        condition_check=SimpleNamespace(input_data_is_fedot_type=lambda data: False),
        automl_config=SimpleNamespace(config={'task': 'classification'}),
        is_finetuned=False,
        solver=None,
    )
    industrial._process_input_data = lambda data: 'processed-train'
    industrial._FedotIndustrial__init_industrial_backend = lambda data: data

    monkeypatch.setattr(
        'fedot.industrial.api.main.build_tuner',
        lambda api, **kwargs: captured.update(kwargs) or 'tuned-model',
    )

    industrial.finetune(
        train_data=('features', 'target'),
        tuning_params={'tuner': 'sequential', 'iterations': 3},
        model_to_tune=_ModelToTune(),
    )

    assert captured['train_data'] == 'processed-train'
    assert captured['tuning_params']['iterations'] == 3
    assert captured['tuning_params']['metric'] == FEDOT_TUNING_METRICS['classification']
    assert captured['tuning_params']['tuner'] == FEDOT_TUNER_STRATEGY['sequential']
    assert industrial.manager.is_finetuned is True
    assert industrial.manager.solver == 'tuned-model'


def test_industrial_main_get_metrics_warns_when_roc_auc_has_no_probabilities(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    logs = []
    industrial.logger = SimpleNamespace(info=lambda message: logs.append(message))
    industrial.manager = SimpleNamespace(automl_config=SimpleNamespace(task='classification'))
    industrial._metric_evaluation_loop = lambda **kwargs: {'roc_auc': 0.7}

    result = industrial.get_metrics(
        labels=np.array([1, 0]),
        probs=None,
        target=np.array([1, 0]),
        metric_names=('roc_auc',),
    )

    assert result == {'roc_auc': 0.7}
    assert logs == ['Predicted probabilities are not available. Use `predict_proba()` method first']


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

    industrial.explain({'method': 'recurrence', 'samples': 3, 'window': 7,
                       'metric': 'f1', 'threshold': 80, 'name': 'demo'})

    assert captured['features'].shape == (2, 3)
    assert np.array_equal(captured['target'], np.array([1, 2]))
    assert captured['explain'] == (3, 7, 'f1')
    assert captured['visual'] == ('f1', 80, 'demo')


def test_industrial_main_load_uses_rule_based_path_resolution(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    industrial.manager = SimpleNamespace()

    monkeypatch.setattr('fedot.industrial.api.main.IndustrialModels',
                        lambda: SimpleNamespace(setup_repository=lambda backend=None: 'repo'))
    monkeypatch.setattr('fedot.industrial.api.main.os.listdir', lambda path: ['pipeline_saved_1', 'fitted_operations'])
    monkeypatch.setattr(
        'fedot.industrial.api.main.Pipeline',
        lambda: SimpleNamespace(
            load=lambda path: f'loaded:{path}'))

    result = industrial.load('root')

    assert result == ['loaded:root/pipeline_saved_1/pipeline_saved_1/0_pipeline_saved',
                      'loaded:root/pipeline_saved_1/fitted_operations/0_pipeline_saved']


def test_industrial_main_save_uses_rule_based_mode_selection(monkeypatch):
    industrial = FedotIndustrial.__new__(FedotIndustrial)
    captured = []

    class FakeManager:
        def __init__(self):
            self.solver = SimpleNamespace(current_pipeline=SimpleNamespace(
                save=lambda **kwargs: captured.append(('model', kwargs))))
            self.compute_config = SimpleNamespace(output_folder='out')
            self.condition_check = SimpleNamespace(solver_is_fedot_class=lambda solver: False)
            self.logger = SimpleNamespace(info=lambda msg: captured.append(('log', msg)))
            self.predicted_labels = np.array([1, 0])

        def create_folder(self, output_folder):
            captured.append(('folder', output_folder))

    industrial.manager = FakeManager()
    industrial.metric_dict = SimpleNamespace(to_csv=lambda path: captured.append(('metrics', path)))

    monkeypatch.setattr(
        'fedot.industrial.api.main.pd.DataFrame', lambda values: SimpleNamespace(
            to_csv=lambda path: captured.append(
                ('prediction', path))))

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
