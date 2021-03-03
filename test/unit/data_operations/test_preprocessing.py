import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression

from fedot.core.data.data import InputData
from fedot.core.data.preprocessing import TextPreprocessingStrategy
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup() -> InputData:
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


@pytest.fixture()
def np_array_regression_with_missing_values() -> np.array:
    np.random.seed(42)
    features, target = make_regression(n_samples=1024, n_features=20, shuffle=True)

    data = pd.DataFrame(features)
    mask = np.random.choice([True, False], p=[0.1, 0.9], size=data.shape)
    data = data.mask(mask).to_numpy()

    return data


# TODO: refactor to DataOperation test
@pytest.mark.skip('TODO: Use the content of the test for DataOperation tests')
def test_scaling_with_imputation(np_array_regression_with_missing_values):
    pass
    # scaler = ScalingWithImputation()
    # actual_scaled_data = scaler.fit_apply(np_array_regression_with_missing_values)
    #
    # scaler = StandardScaler()
    # df = pd.DataFrame(np_array_regression_with_missing_values)
    # data = df.fillna(df.mean()).to_numpy()
    # expected_scaled_data = scaler.fit_transform(data)
    #
    # result = []
    # for i in range(len(actual_scaled_data)):
    #     for j in range(len(actual_scaled_data[0])):
    #         result.append(math.isclose(actual_scaled_data[i][j], expected_scaled_data[i][j], abs_tol=0.00001))
    #
    # assert all(result)


def test_text_preprocessing_strategy():
    test_text = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    preproc_strategy = TextPreprocessingStrategy()

    fit_result = preproc_strategy.fit(test_text)

    apply_result = preproc_strategy.apply(test_text)

    assert isinstance(fit_result, TextPreprocessingStrategy)
    assert apply_result[0] != test_text[0]
