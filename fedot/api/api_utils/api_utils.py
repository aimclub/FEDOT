from typing import Union
import numpy as np
from fedot.core.repository.tasks import Task
import pandas as pd
from fedot.core.data.data import InputData, OutputData
from fedot.api.api_utils.metrics import Fedot_metrics_helper
from fedot.api.api_utils.composer import Fedot_composer_helper
from fedot.api.api_utils.params import Fedot_params_helper
from fedot.api.api_utils.data import Fedot_data_helper


class Api_facade(Fedot_data_helper, Fedot_composer_helper, Fedot_metrics_helper):

    def __init__(self, **input_params):
        self.params_model = Fedot_params_helper()
        self.api_params = self.params_model.initialize_params(**input_params)

    def initialize_params(self):
        return self.api_params

    def save_predict(self, predicted_data: OutputData):
        if len(predicted_data.predict.shape) >= 2:
            prediction = predicted_data.predict.tolist()
        else:
            prediction = predicted_data.predict
        return pd.DataFrame({'Index': predicted_data.idx,
                             'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)
