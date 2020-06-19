from sklearn.svm import LinearSVC as SklearnSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np


class CustomSVC:
    def fit(self, train_data: np.array, target_data: np.array):
        self.fitted_model = SklearnSVC()
        self.classes_ = np.unique(target_data)
        if len(self.classes_) > 2:
            self.fitted_model = CalibratedClassifierCV(self.fitted_model, cv=10)

        self.fitted_model.fit(train_data, target_data)
        return self.fitted_model

    def predict(self, data_to_predict: np.array):
        return self.fitted_model.predict(data_to_predict)
