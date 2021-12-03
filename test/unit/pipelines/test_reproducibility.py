import random

import numpy as np

from fedot.core.data.data_split import train_test_data_setup
from test.integration.quality.test_synthetic_tasks import get_regression_pipeline, get_regression_data


def test_reproducubility():
    np.random.seed(1)
    random.seed(1)

    ref_pipeline = get_regression_pipeline()
    input_data = get_regression_data()

    train_data, test_data = train_test_data_setup(input_data)

    ref_pipeline.fit_from_scratch(train_data)
    pred_1 = ref_pipeline.predict(test_data)

    np.random.seed(1)
    random.seed(1)

    input_data = get_regression_data()
    train_data, test_data = train_test_data_setup(input_data)

    ref_pipeline.unfit()

    ref_pipeline.fit_from_scratch(train_data)
    pred_2 = ref_pipeline.predict(test_data)

    assert np.array_equal(pred_1.predict, pred_2.predict)
