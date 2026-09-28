from fedot.industrial.core.architecture.settings.computational import backend_methods as np


class Anomaly:
    def __init__(self, params: dict):
        self.level = params.get('level', 10)
        self.anomaly_type = self.__class__.__name__

    def get(self, ts: np.ndarray, interval: tuple):
        NotImplementedError()


class ShiftTrendUP(Anomaly):
    def __init__(self, params):
        super().__init__(params)

    def get(self, ts: np.ndarray, interval: tuple):
        shift = np.zeros(ts.size)
        shift_value = np.mean(
            ts[interval[0]:interval[1] + 1]) * (self.level / 100)
        shift_value = abs(shift_value)
        shift[interval[0]:interval[1] + 1] = shift_value
        return self.apply_shift(ts, shift)

    def apply_shift(self, ts: np.ndarray, shift: np.ndarray):
        return ts + shift


class ShiftTrendDOWN(ShiftTrendUP):
    def __init__(self, params):
        super().__init__(params)

    def apply_shift(self, ts: np.ndarray, shift: np.ndarray):
        return ts - shift


class DecreaseDispersion(Anomaly):

    def __init__(self, params):
        super().__init__(params)

    def get(self, ts: np.ndarray, interval: tuple):
        new_ts = ts.copy()
        sector_values = new_ts[interval[0]:interval[1] + 1]
        mean = float(np.mean(sector_values))
        new_sector_values = [self.shrink(mean, x) for x in sector_values]
        new_ts[interval[0]:interval[1] + 1] = new_sector_values
        return new_ts

    def shrink(self, mean_value: float, i: float):
        diff = mean_value - i
        new_diff = diff - diff * (self.level / 100)
        new_i = mean_value - new_diff
        return new_i


class IncreaseDispersion(DecreaseDispersion):

    def __init__(self, params):
        super().__init__(params)

    def shrink(self, mean_value: float, i: float):
        diff = mean_value - i
        new_diff = diff + diff * (self.level / 100)
        new_i = mean_value - new_diff
        return new_i


class AddNoise(Anomaly):
    def __init__(self, params):
        super().__init__(params)
        self.noise_type = params.get('noise_type', np.random.choice(
            ['gaussian', 'uniform', 'laplace']))

    def get(self, ts: np.ndarray, interval: tuple):
        ts_ = ts.copy()
        sector = ts_[interval[0]: interval[1] + 1]

        noise_std = np.std(sector) * self.level / 100

        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, noise_std, len(sector))
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-noise_std, noise_std, len(sector))
        elif self.noise_type == 'laplace':
            noise = np.random.laplace(0, noise_std, len(sector))
        else:
            raise ValueError(
                "Invalid noise_type. Please choose 'gaussian', 'uniform', or 'laplace'.")

        noisy_sector = sector + noise
        ts_[interval[0]:interval[1] + 1] = noisy_sector

        return ts_


class Peak(Anomaly):
    def __init__(self, params):
        super().__init__(params)

    def get(self, ts: np.ndarray, interval: tuple):
        ts_ = ts.copy()
        shift = np.zeros(ts.size)
        sector = ts_[interval[0]: interval[1] + 1]
        peak_value = abs(np.mean(sector) * (self.level / 100))
        center_point = int((interval[1] + 1 + interval[0]) / 2)
        shift[center_point] = peak_value
        return self.apply_shift(ts_, shift)

    def apply_shift(self, ts_, shift):
        return ts_ + shift


class Dip(Peak):
    def __init__(self, params):
        super().__init__(params)

    def apply_shift(self, ts_, shift):
        return ts_ - shift


class ChangeTrend(Anomaly):
    def __init__(self, params):
        super().__init__(params)

    def get(self, ts: np.ndarray, interval: tuple):
        pass
