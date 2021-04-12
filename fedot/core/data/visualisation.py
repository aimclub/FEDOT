import matplotlib.pyplot as plt
import numpy as np


def plot_forecast(pre_history: 'InputData', forecast: 'OutputData'):
    # TODO add docstring description and refactor for preprocessing PR
    last_ind = int(round(pre_history.idx[-1]))
    plt.figure(figsize=(20, 10))
    plt.plot(pre_history.idx[-72:], pre_history.target[-72:])
    ticks = range(last_ind, last_ind + len(forecast.predict) + 1)
    ts = np.append(pre_history.target[-1], forecast.predict)
    plt.plot(ticks, ts)
    plt.show()
