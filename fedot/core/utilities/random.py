import numpy as np

MODEL_FITTING_SEED = 0


class RandomStateHandler:
    def __init__(self, seed: int = 0):
        self._seed = seed
        self._old_seed = None

    def __enter__(self):
        self._old_state = np.random.get_state()
        np.random.seed(self._seed)
        return self._seed

    def __exit__(self, exc_type, exc_value, exc_traceback):
        np.random.set_state(self._old_state)
