import os
import random

import numpy as np
import pandas as pd
import pytest

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.models.data import train_test_data_setup
from core.repository.atomised_models import atomise_chain, read_atomised_model, ModelTypesIdsEnum
from test.test_composer import roc_auc, baseline_chain


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/test_dataset2.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.mark.parametrize('data_fixture', ['file_data_setup'])
def test_atomised_model_is_similar_to_prototype(data_fixture, request):
    random.seed(1)
    np.random.seed(1)
    data = request.getfixturevalue(data_fixture)
    dataset_to_fit, dataset_to_validate = train_test_data_setup(data=data)

    prototype = baseline_chain()
    alt_chain = Chain()
    alt_chain.nodes = [NodeGenerator.primary_node(ModelTypesIdsEnum.xgboost)]

    model_title = 'test_atomised_model'
    atomise_chain(model_name=model_title,
                  chain=prototype)
    atomised_model_id = read_atomised_model(model_name=model_title)
    chain_from_atomised = Chain()
    chain_from_atomised.nodes = [NodeGenerator.primary_node(atomised_model_id)]

    prototype.fit(dataset_to_fit)
    alt_chain.fit(dataset_to_fit)
    chain_from_atomised.fit(dataset_to_fit)

    roc_auc_chain_prototype = roc_auc(y_true=dataset_to_validate.target,
                                      y_score=prototype.predict(dataset_to_validate).predict)
    roc_auc_chain_alt = roc_auc(y_true=dataset_to_validate.target,
                                y_score=alt_chain.predict(dataset_to_validate).predict)
    roc_auc_chain_from_atomised = roc_auc(y_true=dataset_to_validate.target,
                                          y_score=chain_from_atomised.predict(dataset_to_validate).predict)

    # Prediction of the atomised model is the same as for prototype
    assert roc_auc_chain_prototype == roc_auc_chain_from_atomised
    # Check that other chain has different prediction
    assert roc_auc_chain_prototype != roc_auc_chain_alt
