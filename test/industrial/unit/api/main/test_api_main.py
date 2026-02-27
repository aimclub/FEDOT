import os
import warnings
import shutil

import numpy as np
import pytest
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from matplotlib import get_backend, pyplot as plt

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.repository.config_repository import DEFAULT_CLF_AUTOML_CONFIG, DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_REG_AUTOML_CONFIG
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH
from fedot_ind.tools.synthetic.synth_ts_data import SynthTimeSeriesData
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator

from fedot.core.pipelines.pipeline import Pipeline


def univariate_clf_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    return train_data


def univariate_regression_data():
    generator = TimeSeriesDatasetsGenerator(task='regression',
                                            binary=True,
                                            multivariate=False)
    train_data, test_data = generator.generate_data()

    return train_data


def multivariate_clf_data():
    generator = TimeSeriesDatasetsGenerator(task='classification',
                                            binary=True,
                                            multivariate=True)
    train_data, test_data = generator.generate_data()

    return train_data


def multivariate_regression_data():
    generator = TimeSeriesDatasetsGenerator(task='regression',
                                            binary=True,
                                            multivariate=True)
    train_data, test_data = generator.generate_data()

    return train_data


def fedot_industrial_classification():
    AUTOML_LEARNING_STRATEGY = dict(timeout=0.1,
                                    logging_level=50)
    LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                       'optimisation_loss': {'quality_loss': 'f1'}}
    INDUSTRIAL_CONFIG = {'problem': 'classification'}
    API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
                  'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
                  'learning_config': LEARNING_CONFIG,
                  'compute_config': DEFAULT_COMPUTE_CONFIG}
    return FedotIndustrial(**API_CONFIG)


def fedot_industrial_regression():
    AUTOML_LEARNING_STRATEGY = dict(timeout=0.1,
                                    logging_level=50)
    LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                       'optimisation_loss': {'quality_loss': 'rmse'}}
    INDUSTRIAL_CONFIG = {'problem': 'regression'}
    API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
                  'automl_config': DEFAULT_REG_AUTOML_CONFIG,
                  'learning_config': LEARNING_CONFIG,
                  'compute_config': DEFAULT_COMPUTE_CONFIG}
    return FedotIndustrial(**API_CONFIG)


@pytest.mark.parametrize('metric_names, data_func, fedot_func', (
    [('f1', 'accuracy'), univariate_clf_data, fedot_industrial_classification],
    [('f1', 'accuracy'), multivariate_clf_data, fedot_industrial_classification],
    [('rmse', 'mae'), univariate_regression_data, fedot_industrial_regression],
    [('rmse', 'mae'), multivariate_regression_data, fedot_industrial_regression],
), ids=['clf_uni', 'clf_multi', 'reg_uni', 'reg_multi'])
def test_fit_predict_fedot_industrial(metric_names, data_func, fedot_func):
    data = data_func()
    fedot_industrial = fedot_func()
    fedot_industrial.fit(data)
    predict = fedot_industrial.predict(data)
    predict_proba = fedot_industrial.predict_proba(data)
    metrics = fedot_industrial.get_metrics(predict,
                                           predict_proba,
                                           target=data[1],
                                           metric_names=metric_names)

    fedot_industrial.save()

    for file in ['labels.csv', 'metrics.csv', 'optimization_history.json']:
        filepath = os.path.join(PROJECT_PATH, 'results', file)
        assert os.path.isfile(filepath)

    # search for pipeline
    for file in os.listdir(os.path.join(PROJECT_PATH, 'results')):
        if 'pipeline_saved' in file:
            ppl_path = os.path.join(PROJECT_PATH, 'results', file)
            loaded_ppl = fedot_industrial.load(ppl_path)
            break
    assert isinstance(loaded_ppl, Pipeline)

    shutil.rmtree(os.path.join(PROJECT_PATH, 'results'))

    assert predict.shape[0] == data[1].shape[0]
    assert predict_proba.shape[0] == data[1].shape[0]
    assert metrics is not None
    if len(data[1].shape) > 1:
        assert predict.shape[1] == data[1].shape[1]


@pytest.fixture()
def ts_config():
    return dict(random_walk={'ts_type': 'random_walk',
                             'length': 1000,
                             'start_val': 36.6})


def test_generate_ts(ts_config):
    ts = SynthTimeSeriesData(ts_config).generate_ts()

    assert isinstance(ts, np.ndarray)
    assert ts.shape[0] == 1000


@pytest.fixture()
def anomaly_config():
    return {'dip': {'level': 20,
                    'number': 2,
                    'min_anomaly_length': 10,
                    'max_anomaly_length': 20}
            }


def test_generate_anomaly_ts(ts_config, anomaly_config):
    init_synth_ts, mod_synth_ts, synth_inters = SynthTimeSeriesData(anomaly_config).generate_anomaly_ts(ts_config)
    assert len(init_synth_ts) == len(mod_synth_ts)
    for anomaly_type in synth_inters:
        for interval in synth_inters[anomaly_type]:
            ts_range = range(len(init_synth_ts))
            assert interval[0] in ts_range and interval[1] in ts_range


@pytest.mark.parametrize('data_func, fedot_func, node', (
    [univariate_clf_data, fedot_industrial_classification, 'rf'],
    [multivariate_clf_data, fedot_industrial_classification, 'rf'],
    [univariate_regression_data, fedot_industrial_regression, 'treg'],
    [multivariate_regression_data, fedot_industrial_regression, 'treg'],
), ids=['clf_uni', 'clf_multi', 'reg_uni', 'reg_multi'])
def test_finetune(data_func, fedot_func, node):
    data = data_func()
    fedot_industrial = fedot_func()
    fedot_industrial.finetune(train_data=data,
                              model_to_tune=PipelineBuilder().add_node(node),
                              tuning_params={'tuning_timeout': 0.1})
    assert fedot_industrial.manager.solver is not None


def test_plot_methods():
    industrial = fedot_industrial_classification()
    data = univariate_clf_data()
    industrial.fit(data)
    industrial.predict(data)
    industrial.predict_proba(data)

    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
    industrial.explain()
