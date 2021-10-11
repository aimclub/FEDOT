import os
import h2o


class TPOTRegressionSerializationWrapper:
    """ Wrapper to serialize tpot algorithms.
    Can be used for classification, multioutput regression and time series forecasting"""
    def __init__(self, estimators):
        self._estimators = estimators

    def get_estimators(self):
        return self._estimators


class H2OSerializationWrapper:
    """ Wrapper to serialize h2o algorithms.
    Can be used for classification, multioutput regression and time series forecasting.
    Unfortunately there are no support for all types of h2o pipelines (for this version)"""
    def __init__(self, estimators):
        self._estimators = estimators

    @classmethod
    def load_operation(cls, path_global):
        models = []
        for path in os.listdir(path_global):
            path = os.path.join(path_global, path)
            imported_model = h2o.import_mojo(path)
            models.append(imported_model)
        return H2OSerializationWrapper(models)

    def get_estimators(self):
        return self._estimators

    def save_operation(self, path, operation_id):
        path = os.path.join(path, f'h2o_{operation_id}')
        count = 0
        for model in self._estimators:
            model_name = f"{count}.zip"
            model.save_mojo(os.path.join(path, model_name))
            count += 1
        return path
