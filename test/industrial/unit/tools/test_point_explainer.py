import math
import warnings

import pytest
from matplotlib import get_backend, pyplot as plt

from fedot_ind.tools.explain.distances import DistanceTypes
from fedot_ind.tools.explain.explain import PointExplainer
from tests.unit.api.fixtures import get_data_by_task
from tests.unit.api.main.test_api_main import fedot_industrial_classification

distances = DistanceTypes.keys()


@pytest.mark.parametrize('distance, window', [(d, w) for d in distances for w in [0, 30]])
def test_explain(distance, window):
    # switch to non-Gui, preventing plots being displayed
    # suppress UserWarning that agg cannot show plots
    get_backend()
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
    train_data, test_data = get_data_by_task('classification')

    industrial = fedot_industrial_classification()
    industrial.fit(train_data)

    distance = distance
    explainer = PointExplainer(industrial, test_data[0], test_data[1])
    explainer.explain(n_samples=1, window=window, method=distance)
    explainer.visual(threshold=0, name='Custom' + '_' + distance)

    ts_len = test_data[0].shape[1]
    expected_n_parts = math.ceil(ts_len / (window * ts_len // 100)) if window != 0 else ts_len
    assert explainer.scaled_vector.shape[0] == expected_n_parts
