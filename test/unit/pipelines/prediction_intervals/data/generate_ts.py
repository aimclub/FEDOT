# This script generates train ts, test ts for testing prediction intervals

import numpy as np


def synthetic_series(start, end):

    trend = np.array([5 * np.sin(x / 20) + 0.1 * x - 2 * np.sqrt(x) for x in range(start, end)])
    noise = np.random.normal(loc=0, scale=1, size=end - start)

    return trend + noise


ts_train = synthetic_series(0, 200)
ts_test = synthetic_series(200, 220)
np.savetxt("train_ts.csv", ts_train, delimiter=",")
np.savetxt("test_ts.csv", ts_test, delimiter=",")
