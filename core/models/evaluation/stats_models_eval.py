import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA

from core.models.data import InputData, OutputData


def fit_ar(train_data: InputData):
    return AutoReg(train_data.target, lags=[1, 2, 6, 12, 24],
                   exog=train_data.features).fit()


def fit_arima(train_data: InputData):
    return ARIMA(train_data.target, order=(2, 0, 0),
                 exog=train_data.features).fit(disp=0)


def predict_ar(trained_model, predict_data: InputData) -> OutputData:
    start, end = trained_model.nobs, \
                 trained_model.nobs + len(predict_data.target) - 1
    exog, exog_oos = None, predict_data.features

    if trained_model.data.endog is predict_data.target:
        # if train sample used
        start, end = 0, len(predict_data.target)
        exog, exog_oos = predict_data.features, \
                         predict_data.features

    prediction = trained_model.predict(start=start, end=end,
                                       exog=exog, exog_oos=exog_oos)

    diff = len(predict_data.target) - len(prediction)
    if diff != 0:
        prediction = np.append(prediction, prediction[-diff:])

    return prediction


def predict_arima(trained_model, predict_data: InputData) -> OutputData:
    start, end = trained_model.nobs, \
                 trained_model.nobs + len(predict_data.target) - 1
    exog = predict_data.features

    if trained_model.data.endog is predict_data.target:
        # if train sample used
        start, end = 0, len(predict_data.target)

    prediction = trained_model.predict(start=start, end=end,
                                       exog=exog)

    return prediction[0:len(predict_data.target)]
