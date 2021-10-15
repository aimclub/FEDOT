import matplotlib.pyplot as plt
import numpy as np

from fedot.core.composer.metrics import ROCAUC
from fedot.core.data.data import InputData, OutputData


def plot_forecast(pre_history: 'InputData', forecast: 'OutputData'):
    # TODO add docstring description and refactor for preprocessing PR
    last_ind = int(round(pre_history.idx[-1]))
    plt.figure(figsize=(20, 10))
    plt.plot(pre_history.idx[-72:], pre_history.target[-72:])
    ticks = range(last_ind, last_ind + len(forecast.predict) + 1)
    ts = np.append(pre_history.target[-1], forecast.predict)
    plt.plot(ticks, ts)
    plt.show()


def plot_biplot(prediction: OutputData):
    target = prediction.target
    predict = prediction.predict
    plt.figure(figsize=(10, 10))
    plt.scatter(target, predict)
    plt.grid()
    bisect_x = []
    bisect_y = []
    min_coord = min(np.min(target), np.min(predict))
    max_coord = max(np.max(target), np.max(predict))
    bisect_x.append(min_coord)
    bisect_x.append(max_coord)
    bisect_y.append(min_coord)
    bisect_y.append(max_coord)

    plt.plot(bisect_x, bisect_y)
    plt.title("Biplot")
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.show()


def plot_roc_auc(input_data: InputData, prediction: OutputData):
    fpr, tpr, threshold = ROCAUC.roc_curve(input_data, prediction)
    roc_auc = ROCAUC.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc= 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
