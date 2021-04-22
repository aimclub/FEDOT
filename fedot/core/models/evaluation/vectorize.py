from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from fedot.core.data.data import InputData, OutputData
from fedot.core.models.evaluation.evaluation import EvaluationStrategy
from datetime import timedelta


class VectorizeStrategy(EvaluationStrategy):
    __vectorizers_dict = {
        'tfidf': TfidfVectorizer,
        'cntvect': CountVectorizer,
    }

    def __init__(self, model_type, params):
        self.vectorizer = self.__vectorizers_dict.get(model_type)
        super().__init__(model_type, params)

    def fit(self, train_data: InputData):
        features_list = list(train_data.features)

        vectorizer = self.vectorizer().fit(features_list)

        return vectorizer

    def predict(self, trained_model, predict_data: InputData) -> OutputData:
        return trained_model.transform(list(predict_data.features)).toarray()

    def fit_tuned(self, train_data: InputData, iterations: int,
                  max_lead_time: timedelta = timedelta(minutes=5)):
        raise NotImplementedError()
