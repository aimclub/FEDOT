from core.models.data import (
    InputData
)
from core.models.model import (
    Model
)


class EvaluationStrategy:
    def __init__(self, model: Model):
        self.model = model

    def evaluate(self, data: InputData) -> InputData:
        self.model.fit(data=data)
        return self.model.predict(data=data)
