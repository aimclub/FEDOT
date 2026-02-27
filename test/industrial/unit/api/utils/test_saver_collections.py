import os.path
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedot_ind.tools.serialisation.path_lib import DEFAULT_PATH_RESULTS
from fedot_ind.tools.serialisation.saver_collections import ResultSaver

CUSTOM_PATH = './results'


@pytest.fixture
def sample_results():
    labels = [[np.random.randint(0, 5)] for _ in range(10)]
    probs = np.random.rand(10, 5).round(3)
    metrics = {'roc_auc': np.random.rand(1).round(3),
               'f1': np.random.rand(1).round(3),
               'accuracy': np.random.rand(1).round(3)}
    baseline_metrics = {'roc_auc': np.random.rand(1),
                        'f1': np.random.rand(1),
                        'accuracy': np.random.rand(1)}
    return {
        'labels': labels,
        'probs': probs,
        'metrics': metrics,
        'baseline_metrics': baseline_metrics}


@pytest.mark.parametrize('path', [CUSTOM_PATH, DEFAULT_PATH_RESULTS])
def test_init_result_saver(path):
    dataset_name = 'name'
    generator_name = 'generator'
    saver = ResultSaver(dataset_name=dataset_name,
                        generator_name=generator_name, output_dir=path)
    ds_folder = os.path.abspath(os.path.join(
        saver.output_dir, generator_name, dataset_name))
    gen_folder = os.path.abspath(
        os.path.join(saver.output_dir, generator_name))

    assert os.path.abspath(saver.output_dir) == os.path.abspath(path)
    assert os.path.isdir(saver.output_dir)
    assert os.path.isdir(ds_folder)
    assert os.path.isdir(gen_folder)

    # Keep your test folder clean!
    if path != DEFAULT_PATH_RESULTS:
        shutil.rmtree(Path(saver.output_dir))


@pytest.mark.parametrize('prediction_type', ('labels',
                         'probs', 'metrics', 'baseline_metrics'))
def test_save(prediction_type, sample_results):
    results = sample_results
    dataset_name = 'name'
    generator_name = 'generator'
    output_dir = './results'
    expected_file_path = os.path.join(
        output_dir, generator_name, dataset_name, f'{prediction_type}.csv')

    saver = ResultSaver(dataset_name=dataset_name,
                        generator_name=generator_name, output_dir=output_dir)
    saver.save(
        predicted_data=results[prediction_type],
        prediction_type=prediction_type)

    assert os.path.isfile(expected_file_path)
    saved_data = pd.read_csv(expected_file_path, index_col=0)

    if prediction_type in ('metrics', 'baseline_metrics'):
        for m in ['roc_auc', 'f1', 'accuracy']:
            assert saved_data[m][0], results[prediction_type][m][0]
    elif prediction_type in ('labels', 'probs'):
        arr = saved_data.values
        assert np.allclose(arr, results[prediction_type], rtol=1.e-3)
    # Keep your test folder clean!
    shutil.rmtree(Path(saver.output_dir))
