from fedot.api.api_utils.initial_assumptions import API_initial_assumptions_helper
import pandas as pd
from fedot.core.data.data import OutputData
from fedot.api.api_utils.metrics import API_metrics_helper
from fedot.api.api_utils.composer import API_composer_helper
from fedot.api.api_utils.params import API_params_helper
from fedot.api.api_utils.data import API_data_helper


class Api_facade(API_data_helper, API_composer_helper, API_metrics_helper, API_initial_assumptions_helper):

    def __init__(self, **input_params):
        self.composer_params = API_params_helper()
        self.api_params = self.composer_params.initialize_params(**input_params)

    def initialize_params(self):
        return self.api_params

    def save_predict(self, predicted_data: OutputData):
        if len(predicted_data.predict.shape) >= 2:
            prediction = predicted_data.predict.tolist()
        else:
            prediction = predicted_data.predict
        return pd.DataFrame({'Index': predicted_data.idx,
                             'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)
