from typing import Optional

import numpy as np
from h2o import h2o, H2OFrame
from h2o.automl import H2OAutoML

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy


class H2OAutoMLStrategy(EvaluationStrategy):
    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.name_operation = operation_type
        self.params = params
        self.best_model = None
        super().__init__(operation_type)

    def fit(self, train_data: InputData):
        ip, port = self._get_h2o_connect_config()

        h2o.init(ip=ip, port=port, name='h2o_server')

        frame = self._data_transform(train_data)

        train_frame, valid_frame = frame.split_frame(ratios=[0.85])

        # make sure that your target column is the last one
        train_columns = train_frame.columns
        target_name = train_columns[-1]
        train_columns.remove(target_name)
        #train_frame[target_name] = train_frame[target_name].asfactor()
        model = H2OAutoML(max_models=self.params.get("max_models"),
                          seed=self.params.get("seed"),
                          max_runtime_secs=self.params.get("timeout")
                          )

        model.train(x=train_columns, y=target_name, training_frame=train_frame)
        self.best_model = model.leader

        return self.best_model

    def predict(self, trained_operation, predict_data: InputData, is_fit_pipeline_stage: bool) -> OutputData:
        test_frame = self._data_transform(predict_data)

        target_name = test_frame.columns[-1]
        test_frame[target_name] = test_frame[target_name].asfactor()

        prediction_frame = trained_operation.predict(test_frame)

        # return list of values like predict_proba[,:1] in sklearn
        print(prediction_frame)
        prediction: list = prediction_frame['p1'].transpose().getrow()

        return self._convert_to_output(prediction, predict_data)

    def _convert_to_operation(self, operation_type: str):
        pass

    def _data_transform(self, data: InputData) -> H2OFrame:
        concat_data = np.concatenate((data.features, data.target.reshape(-1, 1)), 1)
        frame = H2OFrame(python_obj=concat_data)
        return frame

    def _get_h2o_connect_config(self):
        ip = '127.0.0.1'
        port = 8888
        return ip, port
