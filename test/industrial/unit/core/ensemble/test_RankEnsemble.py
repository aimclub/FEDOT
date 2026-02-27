import os

import pytest

from fedot_ind.core.architecture.postprocessing.results_picker import ResultsPicker
from fedot_ind.core.ensemble.rank_ensembler import RankEnsemble
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH


@pytest.fixture()
def get_proba_metric_dict():
    results_path = os.path.join(
        PROJECT_PATH, 'tests/data/classification_results')
    picker = ResultsPicker(path=results_path)
    proba_dict, metric_dict = picker.run()
    return proba_dict, metric_dict


def test_rank_ensemble_umd(get_proba_metric_dict):
    proba_dict, metric_dict = get_proba_metric_dict

    ensembler_umd = RankEnsemble(dataset_name='UMD',
                                 proba_dict=proba_dict,
                                 metric_dict=metric_dict)
    result = ensembler_umd.ensemble()

    assert result['Base_metric'] == 0.993
    assert result['Base_model'] == 'fedot_preset'


def test__create_models_rank_dict(get_proba_metric_dict):
    proba_dict, metric_dict = get_proba_metric_dict
    ensembler = RankEnsemble(dataset_name='UMD',
                             proba_dict=proba_dict,
                             metric_dict=metric_dict)
    model_rank = ensembler._create_models_rank_dict(
        prediction_proba_dict=proba_dict, metric_dict=metric_dict)
    assert isinstance(model_rank, dict)


def test__sort_models(get_proba_metric_dict):
    proba_dict, metric_dict = get_proba_metric_dict
    ensembler = RankEnsemble(dataset_name='UMD',
                             proba_dict=proba_dict,
                             metric_dict=metric_dict)
    model_rank = ensembler._create_models_rank_dict(
        prediction_proba_dict=proba_dict, metric_dict=metric_dict)
    sorted_dict = ensembler._sort_models(model_rank=model_rank)
    assert isinstance(sorted_dict, dict)
