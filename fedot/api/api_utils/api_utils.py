from fedot.api.api_utils.initial_assumptions import ApiInitialAssumptions
import pandas as pd
from fedot.core.data.data import OutputData
from fedot.api.api_utils.metrics import ApiMetrics
from fedot.api.api_utils.composer import ApiComposer
from fedot.api.api_utils.params import ApiParams
from fedot.api.api_utils.data import ApiDataHelper


class ApiFacade(ApiDataHelper, ApiComposer, ApiMetrics, ApiInitialAssumptions):

    def __init__(self, **input_params):
        self.composer_params = ApiParams()
        self.api_params = self.composer_params.initialize_params(**input_params)

    def initialize_params(self):
        return self.api_params

    def save_predict(self, predicted_data: OutputData):
        if len(predicted_data.predict.shape) >= 2:
            prediction = predicted_data.predict.tolist()
        else:
            prediction = predicted_data.predict
        pd.DataFrame({'Index': predicted_data.idx,
                      'Prediction': prediction}).to_csv(r'./predictions.csv', index=False)
        self.api_params['logger'].info('Predictions was saved in current directory.')
