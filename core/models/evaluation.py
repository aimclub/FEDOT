from core.models.data import (
    Data
)
from core.models.model import (
    Model
)


class EvaluationStrategy:
    def __init__(self, model: Model):
        self.model = model

    def evaluate(self, data: Data) -> Data:
        self.model.fit(data=data)
        return self.model.predict(data=data)


