import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from golem.core.utilities.data_structures import ensure_wrapped_in_sequence

from fedot.core.utils import default_fedot_data_dir, labels_to_dummy_probs, save_file_to_csv


def test_default_fedot_data_dir():
    default_fedot_data_dir()
    assert 'FEDOT' in os.listdir(str(Path("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())))


def test_labels_to_dummy_probs():
    probs = labels_to_dummy_probs(np.array(['ok', 'not ok', 'ok']))
    print(probs)

    assert len(probs) == 3
    assert len(probs[0]) == 2


def test_save_file_to_csv():
    test_file_path = str(os.path.dirname(__file__))
    dataframe = pd.DataFrame(data=[[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [10, 11, 12],
                                   [13, 14, 15]])
    file_to_save = os.path.join(test_file_path, 'test_file_to_save.csv')
    save_file_to_csv(dataframe, file_to_save)

    with open(file_to_save, 'r') as file:
        content = file.readlines()

    assert os.path.exists(file_to_save)
    assert '1,2,3' in content[1]
    os.remove(file_to_save)


def test_ensure_wrapped_in_sequence():
    cases = [
        str(),
        int(),
        (i for i in range(5)),
        [int()],
        (int(),),
    ]

    container_types = [list, tuple, set]

    for case in cases:
        for container_type in container_types:
            assert isinstance(ensure_wrapped_in_sequence(case, container_type), container_type)

    assert ensure_wrapped_in_sequence(None) is None
