from random import randint


class RandomMetric:
    @staticmethod
    def get_value() -> float:
        return randint(0, 1000)
