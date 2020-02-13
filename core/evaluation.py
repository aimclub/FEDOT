from typing import Tuple

from core.data import (
    Data
)
from core.data import split_train_test, normalize
from core.model import (
    Model
)


class EvaluationStrategy:
    def __init__(self, model: Model):
        self.model = model

    def evaluate(self, data: Data) -> Data:
        self.model.fit(data=data)
        return self.model.predict(data=data)


def train_test_data_setup(data: Data) -> Tuple[Data, Data]:
    train_data_x, test_data_x = split_train_test(data.features)
    train_data_y, test_data_y = split_train_test(data.target)
    train_idx, test_idx = split_train_test(data.idx)
    train_data = Data(features=normalize(train_data_x), target=train_data_y,
                      idx=train_idx)
    test_data = Data(features=normalize(test_data_x), target=test_data_y, idx=test_idx)
    return train_data, test_data
