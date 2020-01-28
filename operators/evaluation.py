from abc import ABC, abstractmethod


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self):
        pass


class LogRegression(EvaluationStrategy):
    def __init__(self):
        pass

    def evaluate(self):
        return self.predict()

    def fit(self):
        pass

    def predict(self):
        return 'LogRegPredict'


class LinRegression(EvaluationStrategy):
    def __init__(self):
        pass

    def evaluate(self):
        return self.predict()

    def fit(self):
        pass

    def predict(self):
        return 'LinRegPredict'
