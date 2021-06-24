import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# a_actual = np.array([2, 3.5, 5, 4, 5, 6.2, 7.5])
# b_actual = np.array([1, 2.5, 4, 2, 6, 4.2, 5.5])
# c_actual = np.array([3, 4, 4, 4, 5, 5, 5])
#
# a_preds = np.array([1.5, 4.5, 4, 5, 6, 6.2, 7.0])
# b_preds = np.array([2, 3.5, 3, 30, 500, 3.2, 4.5])
# c_preds = np.array([2, 5, 3, 5, 4, 4, 6])

a_actual = np.random.random(500)
b_actual = np.random.random(500)
c_actual = np.random.random(500)

a_preds = np.random.random(500)
b_preds = np.random.random(500)
c_preds = np.random.random(500)

# Расчет метрики по отдельности
a_metric = mean_absolute_error(a_actual, a_preds)
b_metric = mean_absolute_error(b_actual, b_preds)
c_metric = mean_absolute_error(c_actual, c_preds)

print((a_metric + b_metric + c_metric)/3)

# Расчет метрики одновременно
actuals = np.hstack((a_actual, b_actual, c_actual))
predicted = np.hstack((a_preds, b_preds, c_preds))

print(mean_absolute_error(actuals, predicted))

actual = np.array([2, 5, 3, 5, 4, 4, 6]).reshape(-1, 1)
pred = np.array([2, 5, 3, 5, 4, 4, 6]).reshape(-1, 1)
metric = mean_squared_error(actual, pred, squared=False)
