import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class CustomSVC:
    def __init__(self):
        self.fitted_model = None
        self.classes_ = None

    def fit(self, train_data: np.array, target_data: np.array):
        self.fitted_model = OneVsRestClassifier(SVC(kernel='linear',
                                                    probability=True,
                                                    class_weight='balanced'))
        self.classes_ = np.unique(target_data)
        self.fitted_model.fit(train_data, target_data)
        return self.fitted_model

    def predict(self, data_to_predict: np.array):
        return self.fitted_model.predict(data_to_predict)

    def predict_proba(self, data_to_predict: np.array):
        return self.fitted_model.predict_proba(data_to_predict)

    def get_params(self):
        return self.fitted_model.get_params()
