<<<<<<< HEAD:core/models/evaluation.py
from typing import Tuple

from core.models.data import (
    Data
)
from core.models.data import split_train_test, normalize
from core.models.model import (
=======
from core.data import (
    Data
)
from core.model import (
>>>>>>> chain-roll-down:core/evaluation.py
    Model
)


class EvaluationStrategy:
    def __init__(self, model: Model):
        self.model = model

    def evaluate(self, data: Data) -> Data:
        self.model.fit(data=data)
        return self.model.predict(data=data)


