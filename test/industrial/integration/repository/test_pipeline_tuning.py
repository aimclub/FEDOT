import golem.core.log

from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from golem.core.tuning.sequential import SequentialTuner

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_API_CONFIG
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader


def initialize_uni_data():
    train_data, test_data = DataLoader('Lightning7').load_data()
    train_input_data = init_input_data(train_data[0], train_data[1])
    test_input_data = init_input_data(test_data[0], test_data[1])
    return train_input_data, test_input_data


def initialize_multi_data():
    train_data, test_data = DataLoader('Epilepsy').load_data()
    train_input_data = init_input_data(train_data[0], train_data[1])
    test_input_data = init_input_data(test_data[0], test_data[1])
    return train_input_data, test_input_data


def mock_message(self, msg: str, **kwargs):
    level = 40
    self.log(level, msg, **kwargs)


def test_industrial_uni_series(monkeypatch):
    # monkeypatch golem message function
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)

    IndustrialModels().setup_repository()
    train_data, test_data = DataLoader('Lightning7').load_data()

    api_config = dict(task='classification',
                      timeout=1,
                      n_jobs=-1)
    api_config = ApiConfigCheck().update_config_with_kwargs(DEFAULT_CLF_API_CONFIG,
                                                            **api_config)
    model = FedotIndustrial(**api_config)
    model.fit(train_data)
    labels = model.predict(test_data)
    probs = model.predict_proba(test_data)
    model.get_metrics(labels=labels,
                      probs=probs,
                      target=test_data[1],
                      rounding_order=3,
                      metric_names=('accuracy', 'f1'))


def test_tuner_industrial_uni_series(monkeypatch):
    # monkeypatch golem message function
    monkeypatch.setattr(golem.core.log.LoggerAdapter, 'message', mock_message)

    IndustrialModels().setup_repository()
    train_data, test_data = initialize_uni_data()
    # search_space = SearchSpace(get_industrial_search_space(1))
    pipeline_builder = PipelineBuilder()
    pipeline_builder.add_node('eigen_basis')
    pipeline_builder.add_node('quantile_extractor')
    pipeline_builder.add_node('rf')

    pipeline = pipeline_builder.build()

    pipeline_tuner = TunerBuilder(train_data.task) \
        .with_tuner(SequentialTuner) \
        .with_timeout(2) \
        .with_iterations(2) \
        .build(train_data)

    pipeline = pipeline_tuner.tune(pipeline)

    pipeline.fit(train_data)
    pipeline.predict(test_data)
