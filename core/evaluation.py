from core.data import (
    Data,
    DataStream
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


def train_test_data_setup(data: DataStream):
    train_data_x, test_data_x = split_train_test(data.x)
    train_data_y, _ = split_train_test(data.y)
    return normalize(train_data_x), train_data_y, normalize(test_data_x)
